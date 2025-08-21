import torch
import os
import torch.nn as nn
from argparse import ArgumentParser

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, DynamicArchitecturePlans, ArchitecturePlans
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.architectures.get_network_from_plan import get_network_from_plans
from dynamic_network_architectures.architectures.primus import Primus


import numpy as np
import random
import torch.nn as nn
from nnunetv2.utilities.get_network_via_name import get_network_from_name
from timm.layers import trunc_normal_
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
import torch
from dinov2.train.ssl_meta_arch import SSLMetaArch
from omegaconf import OmegaConf as om
from dinov2.configs import dinov2_default_config

from pathlib import Path

def load_pretrained_checkpoint(pretrained_path):
    pretrained_path = Path(pretrained_path)
    default_cfg = om.create(dinov2_default_config)
    cfg = om.load(pretrained_path.parents[2] / "config.yaml")
    cfg = om.merge(default_cfg, cfg)
    model = SSLMetaArch(cfg)
    state_dict = torch.load(pretrained_path, map_location="cpu")
    model.teacher.load_state_dict(state_dict['teacher'], strict=True)
    teacher_vit = model.teacher.backbone
    network = teacher_vit
    return network

def convert_weights_to_primus_m(pretrained_path):
    primus = get_network_from_name(
        "PrimusM",
        input_channels=1,
        output_channels=2,
        input_patchsize=(160, 160, 160),
        allow_init=True,
        deep_supervision=False,
    )
    network = load_pretrained_checkpoint(pretrained_path)

    converted_params = []
    use_cls = 1 if network.use_cls else 0

    for param_name, param in primus.down_projection.named_parameters():
        converted_params.append("down_projection." + param_name)
        vit_param = network.patch_embed
        for key in param_name.split('.'):
            vit_param = getattr(vit_param, key)
        assert param.data.shape == vit_param.data.shape
        setattr(param, "data", torch.clone(vit_param.data))

    if getattr(primus, "register_tokens", None) is not None:
        assert primus.register_tokens.data.shape == network.register_tokens.data.shape
        primus.register_tokens.data = torch.clone(network.register_tokens.data)
        converted_params.append("register_tokens")

    if primus.eva.pos_embed.data.shape == network.pos_embed.data[:, use_cls:].shape:
        print("y")
        primus.eva.pos_embed.data = torch.clone(network.pos_embed.data[:, use_cls:])
    else:
        interpolated = network._interpolate_pos_encoding_3d(torch.zeros_like(primus.eva.pos_embed.data), 160, 160, 160)
        primus.eva.pos_embed.data = torch.clone(interpolated[:, use_cls:])
    converted_params.append("eva.pos_embed")

    n_block_chunks = len(network.blocks)

    for i, eva_block in enumerate(primus.eva.blocks):
        vit_block = network.blocks[i // n_block_chunks][i]
        eva_param_str = f"eva.blocks.{i}."
        primus_eva_block = primus.eva.blocks[i]
        for param_name, param in eva_block.named_parameters():
            if param_name == "gamma_1":
                converted_params.append(eva_param_str + param_name)
                primus.eva.blocks[i].gamma_1.data = torch.clone(vit_block.ls1.gamma.data)
            elif param_name == "gamma_2":
                converted_params.append(eva_param_str + param_name)
                primus.eva.blocks[i].gamma_2.data = torch.clone(vit_block.ls2.gamma.data)
            elif "attn" in param_name and "attn.proj" not in param_name and "attn.norm" not in param_name:
                converted_params.append(eva_param_str + param_name)
                if "q_proj" in param_name:
                    primus_eva_block.attn.q_proj.weight.data = torch.clone(vit_block.attn.q.weight.data)
                    if primus_eva_block.attn.q_proj.bias is not None:
                        primus_eva_block.attn.q_proj.bias.data = torch.clone(vit_block.attn.q.bias.data)
                if "k_proj" in param_name:
                    primus_eva_block.attn.k_proj.weight.data = torch.clone(vit_block.attn.k.weight.data)
                    if primus_eva_block.attn.k_proj.bias is not None:
                        primus_eva_block.attn.k_proj.bias.data = torch.clone(vit_block.attn.k.bias.data)
                if "v_proj" in param_name:
                    primus_eva_block.attn.v_proj.weight.data = torch.clone(vit_block.attn.v.weight.data)
                    if primus_eva_block.attn.v_proj.bias is not None:
                        primus_eva_block.attn.v_proj.bias.data = torch.clone(vit_block.attn.v.bias.data)

            else:
                converted_params.append(eva_param_str + param_name)
                eva_param = primus_eva_block
                vit_param = vit_block
                for key in param_name.split('.'):
                    eva_param = getattr(eva_param, key)
                    vit_param = getattr(vit_param, key)
                setattr(eva_param, "data", torch.clone(vit_param.data))


    primus.eva.norm.weight.data = torch.clone(network.norm.weight.data)
    primus.eva.norm.bias.data = torch.clone(network.norm.bias.data)
    converted_params.append("eva.norm.weight")
    converted_params.append("eva.norm.bias")
            
    diff = set([x[0] for x in primus.named_parameters()]) - set(converted_params)
    assert all(["up_projection" in x for x in list(diff)])

    return primus

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--src', required=True, type=str)
    parser.add_argument('--out', required=True, type=str)
    args = parser.parse_args()

    configuration_plan = ConfigurationPlan(
        data_identifier="nnsslPlans_onemmiso",
        preprocessor_name="DefaultPreprocessor",
        spacing_style="onemmiso",
        normalization_schemes=["ZScoreNormalization"],
        use_mask_for_norm=False,
        resampling_fn_data="resample_data_or_seg_to_shape",
        resampling_fn_data_kwargs={"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
        resampling_fn_mask="resample_data_or_seg_to_shape",
        resampling_fn_mask_kwargs={"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
        spacing=[1, 1, 1],
        patch_size=(128, 128, 128),
    )

    plan = Plan(
        dataset_name="Dataset745_OpenNeuro",
        plans_name="nnsslPlans",
        original_median_spacing_after_transp=[1.0, 1.0, 1.0],  # Arbitrary
        image_reader_writer="NibabelReaderWriter",
        transpose_forward=[0, 1, 2],
        transpose_backward=[0, 1, 2],
        experiment_planner_used="ExperimentPlanner",
        configurations=configuration_plan,  # We just want to save the used configuration plan!
    )

    # Provide some more infos on the pre-training spacing and patch size
    # and the architecture name
    arch_plans = ArchitecturePlans(
        arch_class_name="PrimusM",
        arch_kwargs={},#arch_kwargs,
    )
    adaptation_plan = AdaptationPlan(
        architecture_plans=arch_plans,
        pretrain_plan=plan,
        pretrain_num_input_channels=1,
        recommended_downstream_patchsize=(160, 160, 160),
        key_to_encoder="eva",
        key_to_stem="down_projection",
        keys_to_in_proj=["down_projection.proj"],
        key_to_lpe="eva.pos_embed"
    )
    serialized_adaptation_plan = adaptation_plan.serialize()

    serialized_adaptation_plan["pretrain_plan"]["configurations"] = {
        "key" : serialized_adaptation_plan["pretrain_plan"]["configurations"]
    }

    primus = convert_weights_to_primus_m(args.src)
    state_dict = primus.state_dict()
    final_state_dict = {"network_weights" : state_dict}
    final_state_dict['nnssl_adaptation_plan'] = serialized_adaptation_plan
    final_state_dict['nnssl_adaptation_plan']['pretrain_patch_size'] = (160, 160, 160)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    torch.save(final_state_dict, args.out)
    