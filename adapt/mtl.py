# it is equall to m3 for adapt and m1 for evaluate accordinc to https://docs.google.com/spreadsheets/d/1kk7dfx7HUxDDqfUniEHJPMvnJSwaVaCv8fW4S-1ol_Q/edit?gid=2010450293#gid=2010450293
# tab [Summary] MultiLayers - V21 
import copy
from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from utils.misc import load_templates_from_yaml, print_clip_parameters, print_optimizer_parameters
REFERENCE_TEMPLATE = 'a photo of a {}'


class MTL:
    """
    MTL
    """

    def __init__(self, backbone, lr, classes, alpha_cls=0.0, steps=10,
                 temp_dir='templates.yaml', average_type='loss', interpolate=False,
                 device='cpu', arch='reduced', attn_strategy='naclip', gaussian_std=5.,):
        """
        Initializes the TENT module.

        Args:
            backbone: The CLIP model to be adapted.
            lr: Learning rate for the optimizer.
            steps: Number of steps to adapt.
            device: The device to run the model on (e.g., 'cpu' or 'cuda').

        """

        # loading the base model
        base_model, _ = clip.load(backbone, device)
        self.model = base_model
        self.model.visual.set_params(arch, attn_strategy, gaussian_std)

        self.alpha_cls = alpha_cls

        self.lr = lr
        self.type = type
        self.steps = steps
        self.device = device
        self.interpolate = interpolate

        if temp_dir != 'None':
            # Load the text templates
            self.all_templates = load_templates_from_yaml(temp_dir)
            # print the number of templates
            print(f"Number of templates: {len(self.all_templates)}")
        else:
            self.all_templates = [REFERENCE_TEMPLATE]

        
        assert average_type in ['loss', 'text'], "average_type should be either on 'loss' or 'text'"
        self.average_type = average_type

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

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")

        # extracting text features
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(self.classes,  self.all_templates, average=True).squeeze() # (class, 512)

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
    def evaluate(self, x, classes, vision_outputs=(-1,)):
        """
        Forward pass without adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        Returns:
            pred: Predicted class labels for the input images.

        """
        logits, _, _ = self.model(x, self.text_x[-1], True, vision_outputs=vision_outputs, interpolate=True, vision_out_type="entropy_weighted", save_weights=True) # (#template, batch_size, #classes, H, W)
        logits = logits[0]
        # logits = nn.functional.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

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
        loss_report = []
        for iter in range(self.steps):
            if self.average_type == 'loss':
                # TODO: add a flag to also consider average of text embeddings too
                logits, _, _, cls_logits = self.model(x, self.text_x[:-1], True, vision_outputs=vision_outputs, interpolate=self.interpolate, return_vanilla_cls=True, vision_out_type="mean")   # (#templates, batch_size, #class, W, H)
                
                # adapt
                entropy_per_pixel = self.softmax_entropy(logits)  # Shape: (#template, batch_size, H, W)
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                # Average over all templates, pixels and batch samples
                loss = entropy_per_pixel.mean() + self.alpha_cls * entropy_per_cls.mean()

                # loss = entropy_per_pixel.mean()

                # # average over pixels and batch samples and sum over templates
                # loss = entropy_per_pixel.mean(dim=[-1, -2, -3]).sum()


            elif self.average_type == 'text': #TODO: fix this
                logits, _, _ = self.model(x, self.text_x[-1], True, vision_outputs=vision_outputs, interpolate=False) # (1, batch_size, #classes, H, W)
                entropy_per_pixel = self.softmax_entropy(logits)  # Shape: (batch_size, H, W)
                # Average over all pixels and batch samples
                loss = entropy_per_pixel.mean()

            else:
                raise Exception("average_type should be either on 'loss' or 'text'")
        

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
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

