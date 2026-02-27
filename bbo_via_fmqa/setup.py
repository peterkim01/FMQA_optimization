from setuptools import setup

setup(
    name="bbo_via_fmqa",
    version="0.1.0",
    description="A package for Black-Box Optimization via FMQA",
    author="Woosik Kim, Albert Lee",
    author_email="kim3124@purdue.edu",
    py_modules=[
        "FM_surrogate",
        "read_grid",
        "ising_machine",
        "fmqa_simulated",
        "fmqa_simulated_3D",
        "adaptive_fmqa_simulated",
        "qhd_grid_generator",
        "tester",
    ],
    install_requires=[
        "numpy>=1.21.0",
        "dimod>=0.12.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "fmqa>=0.0.2",
    ],
    # extras_require={
    #     "qci": [
    #         "qci-client>=0.1.0",
    #         "eqc-models>=0.14.1",
    #     ],
    # ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "fmqa-run-batch=tester:main",
            "fmqa-generate-grid=qhd_grid_generator:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
