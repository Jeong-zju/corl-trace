# Third-party notices

Original TRACE contributions use [Apache-2.0](LICENSE). Existing notices in
adapted or vendored source files are retained. This license does not relicense
external datasets, model weights, simulator assets, or hardware SDKs.

| Component | Location / use | Upstream and license |
| --- | --- | --- |
| LeRobot ACT | `policy/lerobot_policy_streaming_act/` configuration, processor and action backbone | [Hugging Face LeRobot](https://github.com/huggingface/lerobot), Apache-2.0; Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team |
| LeRobot Diffusion Policy | `policy/lerobot_policy_prism_diffusion/` configuration, processor and diffusion backbone | [Hugging Face LeRobot](https://github.com/huggingface/lerobot), Apache-2.0; Copyright 2024 Columbia Artificial Intelligence, Robotics Lab, and The HuggingFace Inc. team |
| Signatory | Vendored under `depends/signatory/`; differentiable path signatures | [patrick-kidger/signatory](https://github.com/patrick-kidger/signatory), [Apache-2.0](depends/signatory/LICENSE); Copyright 2019 Patrick Kidger |

TRACE adaptations add causal prefix handling, signature-routed slot updates and
policy conditioning to the upstream policy components. The public source also
specializes these components to the complete TRACE method. Upstream copyright
headers remain in the adapted files.

Runtime dependencies listed in `requirements.txt` are distributed by their
respective maintainers under their own licenses. Optional simulator integrations
expect separately installed environments and assets; those assets are not bundled.

The paper teaser in `docs/assets/teaser.png` comes from the TRACE project materials.
Citation metadata identifies the research authors; it does not change any
institutional ownership rights.
