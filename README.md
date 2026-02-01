# SPACE-CLIP

## Abstract
Contrastive Language-Image Pre-training (CLIP) has accomplished extraordinary success for semantic understanding but inherently struggle to perceive geometric structure. Existing methods attempt to bridge this gap by querying CLIP with textual prompts, a process that is often indirect and inefficient. This paper introduces a fundamentally different approach using dual-pathway decoder. We present SPACE-CLIP, an architecture that unlocks and interprets latent geometric knowledge directly from a frozen CLIP vision encoder, completely bypassing the text encoder and its associated textual prompts. A semantic pathway interprets high-level features, dynamically conditioned on global context using feature-wise linear modulation (FiLM). In addition, a structural pathway extracts fine-grained spatial details from early layers. These complementary streams are hierarchically fused, enabling a robust synthesis of semantic context and precise geometry. Extensive experiments on the KITTI benchmark show that SPACE-CLIP dramatically outperforms previous CLIP-based methods. Our ablation studies validate that
the synergistic fusion of our dual pathways is critical to this success. SPACE-CLIP offers a new, efficient, and architecturally elegant blueprint for repurposing large-scale vision models. The proposed method is
not just a standalone depth estimator, but a readily integrable spatial perception module for the next generation of embodied AI systems, such as vision-language-action (VLA) models.


## Overview

![](https://github.com/user-attachments/assets/a0f91288-003d-476d-8ba1-adecf243fcf5)

The VLA Integration Challenge and the SPACE-CLIP Solution. This figure illustrates the architectural paradigm shift proposed by our work. (a) The Inefficient Path: Conventional CLIP-based methods integrate depth estimation as an external module, creating two major bottlenecks for a Vision-Language-Action (VLA) model. (1) Architectural Conflict arises from needing a separate, inefficient data path from the VLA's primary vision encoder. (2) Textual Interference occurs as the VLA's reasoning engine must generate a textual query, conflicting with its main language tasks. (b) SPACE-CLIP's Seamless Integration: Our model acts as a lightweight, decoder-only module that attaches directly to the VLA's existing, frozen vision encoder. By directly interpreting hierarchical features, it provides spatial awareness without architectural redundancy or textual interference, enabling a truly integrated and efficient agent.

## Architecture

![](https://github.com/user-attachments/assets/4f186d53-a833-442f-bcc6-46b00c6bb986)

The overall architecture of SPACE-CLIP. Our model leverages a frozen CLIP vision encoder to extract multi-level features, which are processed through a novel Dual Pathway architecture within the Dense Predictor. (1) The Semantic Decoder (top path) processes high-level features, dynamically modulated by the global context via a FiLM layer. (2) The Structural Decoder (bottom path) processes low-level features to preserve fine-grained details. The outputs of these two pathways are hierarchically fused at each upsampling stage. This fusion mechanism allows SPACE-CLIP to generate semantically coherent and geometrically precise depth maps.

## Experiment

![](https://github.com/user-attachments/assets/cff55a05-c341-4ad5-8c00-03877bca656a)

![](https://github.com/user-attachments/assets/56a23179-b44e-4f7a-9a3d-28bb72fda968)

![](https://github.com/user-attachments/assets/9c3b3280-ec18-4fea-88c0-74cb51477618)

Qualitative comparison on the KITTI dataset. From left to right: Input Image, Ground Truth, and our SPACE-CLIP's prediction. Our model generates dense and detailed depth maps that accurately capture complex structures such as thin poles, distant vehicles, and foliage, demonstrating its superior geometric understanding.

## Conclusion

SPACE-CLIP offers more than just a performance improvement; it presents a new, efficient, and architecturally elegant paradigm for repurposing foundation models. By keeping the massive pre-trained encoder untouched, our approach promotes a more sustainable and accessible methodology for developing advanced AI systems. Looking forward, we believe this direct interpretation strategy holds immense potential for enriching the next generation of embodied AI, such as VLA models, with the critical spatial awareness needed to intelligently interact with the physical world.


## setup
```bash
conda create -n space-clip python=3.10
conda activate space-clip
pip install -r requirements.txt
```

## Citation
If you find this repository helpful, please consider citing:

```
@misc{cho2026spaceclipspatialperceptionadaptive,
      title={SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation}, 
      author={Taewan Cho and Taeryang Kim and Andrew Jaeyong Choi},
      year={2026},
      eprint={2601.17657},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.17657}, 
}
```

