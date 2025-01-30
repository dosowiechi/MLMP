import copy
from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
# from transformers import AutoProcessor, AutoModel

from utils.misc import load_templates_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_TEMPLATE = 'a photo of a {}'


class WATT:

    def __init__(self, backbone, lr, l=2, m=5, temperature=100,
                 temp_dir='templates.yaml', interpolate=False,
                 device='cpu', arch='reduced', attn_strategy='naclip', gaussian_std=5.,):
        # loading the base model
        base_model, _ = clip.load(backbone, device)
        self.model = base_model
        self.model.visual.set_params(arch, attn_strategy, gaussian_std)

        self.lr = lr
        self.type = type
        self.l = l
        self.m = m
        self.device = device
        self.interpolate = interpolate
        self.temperature = temperature

        if temp_dir != 'None':
            # Load the text templates
            self.all_templates = load_templates_from_yaml(temp_dir)
            # print the number of templates
            print(f"Number of templates: {len(self.all_templates)}")
        else:
            self.all_templates = [REFERENCE_TEMPLATE]

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

    def adapt(self, x, classes, vision_outputs=(-1,)):
        """
        Forward pass with adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        """

        self.reset()
        loss_report = self.perform_adaptation(x, classes, vision_outputs=vision_outputs)
        return loss_report

    @torch.no_grad()
    def evaluate(self, x, classes,  vision_outputs=(-1,)):
        """
        Forward pass without adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        Returns:
            pred: Predicted class labels for the input images.

        """
        text_features = self.extract_text_embeddings(classes, self.all_templates, average=True)
        logits, _, _ = self.model(x, text_features[-1], True, vision_outputs=vision_outputs,
                                  interpolate=True)  # (#template, batch_size, #classes, H, W)
        logits = logits[0]

        return logits

    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)

    def perform_adaptation(self, x, classes, vision_outputs=(-1,)):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """

        with torch.no_grad():
            text_x = self.extract_text_embeddings(classes, self.all_templates, average=False)
        loss_report = []
        for m in range(self.m):
            all_weights = []
            if m == 0:
                self.load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
            else:
                self.model.load_state_dict(avg_state_dict, strict=False)

            loss_report_temp = []
            for text_feat in text_x:
                for l in range(self.l):
                    with torch.no_grad():
                       similarity, _, _ = self.model(x,  text_feat, True, interpolate=self.interpolate)
                       values, pred = similarity[0].topk(1, 1, True, True)
                       pred_flatten = pred.view(-1, 1)
                       pred_inputs = torch.cat([text_feat[c,] for c in pred_flatten]).to(self.device)

                    # Calculating the Loss
                    logits, image_features, text_features = self.model(x,  pred_inputs, True, interpolate=self.interpolate)
                    # logits, _, _ = self.model(x,  text_feat, True, interpolate=self.interpolate)
                    # loss = self.softmax_entropy(logits).mean()
                    image_features = image_features[:, 1:]
                    image_features = image_features.reshape(-1, image_features.shape[-1])
                    images_similarity = image_features @ image_features.t()
                    text_features = text_features.squeeze()
                    texts_similarity = text_features @ text_features.t()
                    targets = F.softmax(self.temperature * ((images_similarity + texts_similarity) / 2), dim=-1)
                    logits = logits.reshape(-1, logits.shape[-3])
                    loss = self.cross_entropy(logits, targets, reduction='mean')

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                loss_report_temp.append(loss.item())

                weights = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        for nparam, p in module.named_parameters():
                            if nparam in ['weight', 'bias']:
                                weights[f"{name}.{nparam}"] = copy.deepcopy(p)
                all_weights.append(weights)

            # Average loss over number of templates for each iteration (total is m)
            loss_report.append(np.mean(loss_report_temp))

            avg_state_dict = self.weight_average(all_weights)

        self.model.load_state_dict(avg_state_dict, strict=False)

        return loss_report

    def extract_text_embeddings(self, class_names, templates, average=True):
        """
        Extracts text embeddings for given class names and templates.

        Args:
            class_names: List of class names to generate text embeddings for.
            templates: List of text templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and templates.
        """
        text_features = []
        for class_name in class_names:
            texts = [template.format(class_name) for template in templates]
            texts = clip.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)  # Shape: (8, 512)
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
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits.
            x : torch.Tensor : logits of shape (#templates, batch_size, num_classes, H, W)
        """
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

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