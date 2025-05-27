import time
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class CLIPARTT:
    """
    CLIPArTT adaptation for open-vocabulary semantic segmentation (OVSS) models.

    Inspired by the original implementation (https://github.com/dosowiechi/CLIPArTT), this method
    selects the top-K predicted classes per pixel to construct refined text prompts and applies
    a self-distillation loss to optimize only the visual encoder’s LayerNorm parameters at test time.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, clipartt_k=3, steps=10, 
                 prompt_dir=None, runtime_calculation=False, 
                 device='cpu',
                 ):

        """
        Args:
            ovss_type (str): Identifier for the open-vocabulary segmentation model to load.
            ovss_backbone (str): Name of the backbone architecture within the OVSS model.
            lr (float): Learning rate for the LayerNorm optimizer.
            classes (List[str]): List of class names for prompt generation.
            clipartt_k (int, optional): Number of top predictions to include in refined prompts. Defaults to 3.
            steps (int, optional): Number of adaptation iterations per sample. Defaults to 10.
            prompt_dir (str or None, optional): Path to YAML file with prompt templates. Defaults to None.
            runtime_calculation (bool, optional): Whether to record adaptation/evaluation runtimes. Defaults to False.
            device (str, optional): Compute device, e.g., 'cpu' or 'cuda'. Defaults to 'cpu'.
        """

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")
        
        self.steps = steps
        self.k = clipartt_k
        self.prompt_dir = prompt_dir
        self.runtime = runtime_calculation
        self.device = device

        # Load the OVSS model and tokenizer
        self.model, self.tokenize = load_ovss(self.ovss_type, self.ovss_backbone, device=self.device)

        if self.prompt_dir:
            # Load the prompt templates
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            # print the number of prompt templates
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # Set the gradients for LayerNorm layers only for visual encoder
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)

        self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters
        params, _ = self.collect_ln_params(self.model.visual)

        # print the parameters
        print_clip_parameters(self.model)

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        print_optimizer_parameters(self.optimizer, self.model)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)


        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(self.classes, self.prompt_templates,
                                                       average=True).squeeze()  # (class, 512)
            
        # define variables to store adaptation and evaluation duration
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    
    def adapt(self, x):
        """
        Forward pass with adaptation.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            List[float]: Loss values recorded at each adaptation iteration.
        """

        self.reset()
        loss_report = self.perform_adaptation(x, self.classes)
        return loss_report

    @torch.no_grad()
    def evaluate(self, x):
        """
        Forward pass without adaptation.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Per-class logits of shape (batch_size, num_classes, H, W).

        """

        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x[-1], True, 
                                  interpolate=True)  # (#template, batch_size, #classes, H, W)
        logits = logits[0]
        t2 = time.time()
        if self.runtime:
            self.eval_times.append(t2-t1)

        return logits

    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)

    def perform_adaptation(self, x, classes):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).
        
        Returns:
            List[float]: Recorded loss values for each adaptation iteration.
        """

        t1 = time.time()
        text_feat = self.extract_text_embeddings(classes, [REFERENCE_PROMPT], average=False).squeeze()
        loss_report = []
        for _ in range(self.steps):
            with torch.no_grad():
                similarity, _, _ = self.model(x,  text_feat, True, interpolate=False)

           # similarity = (100.0 * image_features @ text_feat.T).softmax(dim=-1)
            values, pred = similarity[0].topk(self.k, 1, True, True)
            pred_flatten = pred.view(-1, self.k)
            pred_inputs = torch.cat([self.tokenize(self.getprompt(self.k, c, classes)) for c in pred_flatten]).to(self.device)
            # pred_inputs = pred_inputs.view(pred.shape[0],-1,pred.shape[2],pred.shape[3])

            # Calculating the Loss
            # cosine similarity as logits
            logits, image_features, text_features = self.model(x,  pred_inputs, False, interpolate=False)
            image_features = image_features[:, 1:]
            image_features = image_features.reshape(-1, image_features.shape[-1])
            images_similarity = image_features @ image_features.t()
            text_features = text_features.squeeze()
            texts_similarity = text_features @ text_features.t()
            targets = F.softmax(  ###
                ((images_similarity + texts_similarity) / 2) / 0.01, dim=-1
            )
            logits = logits.reshape(-1, logits.shape[-3])
            loss = self.cross_entropy(logits, targets,
                                      reduction='mean')  ###

            loss_report.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        t2 = time.time()
        if self.runtime:
            self.adapt_times.append(t2-t1)

        return loss_report

    def extract_text_embeddings(self, class_names, prompts, average=True):
        """
        Extracts text embeddings for given class names and prompts.

        Args:
            class_names: List of class names to generate text embeddings for.
            prompts: List of prompt templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different prompts for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and prompts.
        """
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)  # Shape: (#templates, 512)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)  # Shape: (512,)
                class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()
                # add the averaged embeddings to the original embeddings
                class_embeddings = torch.cat([class_embeddings, class_embeddings_avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features

    @staticmethod
    def set_ln_grads(model):
        """
        Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.

        Args:
            model: The model whose LayerNorm layers' gradients are to be set.

        Returns:
            The model with modified gradient settings.
        """
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        """
        Collect the affine scale and shift parameters from LayerNorm layers.

        Args:
            model: The model from which to collect LayerNorm parameters.

        Returns:
            params: List of LayerNorm parameters.
            names: List of parameter names.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names

    @staticmethod
    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    @staticmethod
    def getprompt(K, c, classes):
        for k in range(K):
            if k == 0:
                text_prompt = f"a photo of a " + classes[c[k]]
            else:
                text_prompt = text_prompt + " or " + classes[c[k]]
        return text_prompt

    @staticmethod
    def weight_average(all_weights):
        """
        Compute the average of the weights from multiple models.

        Args:
            all_weights: List of state dictionaries from different models.

        Returns:
            avg_state_dict: Averaged state dictionary.
        """
        K = len(all_weights)
        avg_state_dict = OrderedDict()
        for param_name, param in all_weights[0].items():
            avg_param = sum(sd[param_name] for sd in all_weights) / K
            avg_state_dict[param_name] = avg_param
        return avg_state_dict

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.

        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.

        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """
        Restore the model and optimizer states from copies.

        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)