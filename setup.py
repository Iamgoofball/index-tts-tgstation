
import platform
import os
from setuptools import find_packages, setup

# add fused `anti_alias_activation` cuda extension if CUDA is available
anti_alias_activation_cuda_ext = None
if  platform.system() != "Darwin":
    try:
        from torch.utils import cpp_extension
        if cpp_extension.CUDA_HOME is not None:
            anti_alias_activation_cuda_ext = cpp_extension.CUDAExtension(
                name="indextts.BigVGAN.alias_free_activation.cuda.anti_alias_activation_cuda",
                sources=[
                    "indextts/BigVGAN/alias_free_activation/cuda/anti_alias_activation.cpp",
                    "indextts/BigVGAN/alias_free_activation/cuda/anti_alias_activation_cuda.cu",
                ],
                include_dirs=["indextts/BigVGAN/alias_free_activation/cuda"],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                    ],
                },
            )
        else:
            print("CUDA_HOME is not set. Skipping anti_alias_activation CUDA extension.")
    except ImportError:
        print("PyTorch is not installed. Skipping torch extension.")

setup(
    name="indextts",
    version="0.1.4",
    author="Index SpeechTeam",
    author_email="xuanwu@bilibili.com",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    description="An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System",
    url="https://github.com/index-tts/index-tts",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.6.0",
        "torchaudio",
        "transformers==4.48.3",
        "accelerate",
        "tokenizers",
        "einops",
        "matplotlib",
        "omegaconf",
        "sentencepiece",
        "librosa",
        "numpy",
        "deepspeed",
        "flask",
        "waitress",
        "deepspeed",
        "pysbd",
        "pydub",
        "blake3",
        "wetext",
    ],
    extras_require={
        "webui": ["gradio"],
    },
    ext_modules=[anti_alias_activation_cuda_ext] if anti_alias_activation_cuda_ext else [],
    cmdclass={"build_ext": cpp_extension.BuildExtension} if anti_alias_activation_cuda_ext else {},
    entry_points={
        "console_scripts": [
            "indextts = indextts.cli:main",
        ]
    },
    license="Apache-2.0",
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)