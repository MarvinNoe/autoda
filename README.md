# autoda

## Description
autoda is a plugin-based system designed to create deep learning pipelines. Plugins for performing k-fold cross-validation training, Fast AutoAugment, and SmartAugment optimization are offered. Additionally, an SSD model and the GC10-DET dataset based on PyTorch are available. The dataset can be augmented with Fast AutoAugment, SmartAugment, and TrivialAugment.

## Development

This project was developed using Python 3.9. It is recommended to use Python 3.9 to avoid compatibility issues.

```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Used Libraries and Licenses
This project uses the following libraries:

- `typing_extensions` (License: Python Software Foundation License)
- `pyyaml` (License: MIT License)
- `pillow` (License: Historical Permission Notice and Disclaimer)
- `torch` (License: BSD-3 License)
- `torchvision` (License: BSD-3 License)
- `urllib3` (License: MIT License)
- `pandas` (License: BSD-3 License)
- `ray` (License: Apache 2.0 License)
- `ray[tune]` (License: Apache 2.0 License)
- `torchmetrics[detection]` (License: Apache 2.0 License)
- `hyperopt` (License: BSD-3 License)
- `scikit-learn` (License: BSD-3 License)
- `matplotlib` (License: Python Software Foundation License)
- `kaggle` (License: Apache 2.0 License)

See the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) file for detailed information on the licenses of the used libraries.
