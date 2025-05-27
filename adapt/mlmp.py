import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class MLMP:
    """
    Multi-Layer Multi-Prompt adaptation for open-vocabulary semantic segmentation (OVSS) models.

    Performs test-time adaptation by updating only the visual encoder LayerNorm parameters via 
    multiple prompt and multiple level optimization. 
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, vision_outputs=(-1,), 
                 alpha_cls=0.0, steps=10, prompt_dir='prompts.yaml', 
                 prompt_integration='loss', runtime_calculation=False,
                 device='cpu',
                 ):
        """
        Initialize the MLMP adaptation module.

        Args:
            ovss_type (str): Identifier for the open-vocabulary segmentation model to load.
            ovss_backbone (str): Name of the backbone architecture within the OVSS model.
            lr (float): Learning rate for the LayerNorm optimizer.
            classes (List[str]): List of class names used to generate text embeddings.
            alpha_cls (float, optional): Weight for the classification‐entropy term. Defaults to 0.0.
            steps (int, optional): Number of test-time adaptation iterations per sample. Defaults to 10.
            prompt_dir (str, optional): Path to the YAML file containing prompt templates. Defaults to 'prompts.yaml'.
            prompt_integration (str, optional): Integration mode for prompts, either 'loss' or 'text'. Defaults to 'loss'.
            runtime_calculation (bool, optional): Whether to track adaptation/evaluation runtimes. Defaults to False.
            device (str, optional): Compute device, e.g., 'cpu' or 'cuda'. Defaults to 'cpu'.
        """

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")
        
        self.vision_outputs = vision_outputs
        print(f"+++ The output layers from vision encoder that will be used: {self.vision_outputs}")

        self.alpha_cls = alpha_cls
        self.steps = steps
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

        
        assert prompt_integration in ['loss', 'text'], "prompt_integration should be either on 'loss' or 'text'"
        self.prompt_integration = prompt_integration

        # Set the gradients for LayerNorm layers only for visual encoder
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)

        self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters
        params, _ = self.collect_ln_params(self.model.visual)

        # print the parameters
        print_clip_parameters(self.model)

        # Set the optimizer
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # print the parameters passed to the optimizer
        print_optimizer_parameters(self.optimizer, self.model)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        # extracting text features
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(self.classes,  self.prompt_templates, average=True).squeeze() # (class, 512)

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
        loss_report = self.perform_adaptation(x)
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
        logits, _, _ = self.model(x, self.text_x[-1], True, vision_outputs=self.vision_outputs, 
                                  interpolate=True, vision_out_type="adaptive_weighted_mean", 
                                  save_weights=True) # (#template, batch_size, #classes, H, W)
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

    def perform_adaptation(self, x):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).
        
        Returns:
            List[float]: Recorded loss values for each adaptation iteration.
        """

        t1 = time.time()
        loss_report = []
        for iter in range(self.steps):
            if self.prompt_integration == 'loss':
                logits, _, _, cls_logits = self.model(x, self.text_x[:-1], True, interpolate=False,
                                                      vision_outputs=self.vision_outputs, return_vanilla_cls=True, 
                                                      vision_out_type="mean") # (#templates, batch_size, #class, W, H)
                
                # adapt
                entropy_per_pixel = self.softmax_entropy(logits)  # Shape: (#template, batch_size, H, W)
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                
                # Average over all prompt templates, pixels and batch samples
                loss = entropy_per_pixel.mean() + self.alpha_cls * entropy_per_cls.mean()


            elif self.prompt_integration == 'text':
                logits, _, _ = self.model(x, self.text_x[-1], True, interpolate=False,
                                         vision_outputs=self.vision_outputs) # (1, batch_size, #classes, H, W)                                         
                entropy_per_pixel = self.softmax_entropy(logits)  # Shape: (batch_size, H, W)

                # Average over all pixels and batch samples
                loss = entropy_per_pixel.mean()

            else:
                raise Exception("prompt_integration should be either on 'loss' or 'text'")
        
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
            average: Boolean indicating whether to average the embeddings of different prompt templates for each class.

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

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        """Entropy of softmax distribution from logits.
            x : torch.Tensor : logits of shape (#templates, batch_size, num_classes, H, W)
        """
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

