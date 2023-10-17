# Master Thesis: Multi-task learning for holistic data-driven models of engineering systems

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://github.com/QsingularityAi/Master_Thesis.git
git branch -M main
git push -uf origin main
```

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📦 Features](#-features)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
    - [🔧 Installation](#-installation)
    - [🤖 Running Master_Thesis.git](#-running-Master_Thesis.git)
    - [🧪 Tests](#-tests)
- [🛣 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

## 📍 Overview

## Introduction : 
Multiple linked tasks can now be learned simultaneously thanks to the growing popularity of Multi-Task Learning (MTL) in many deep learning applications. This has led the way for novel modelling techniques. MTL has been included into engineering sciences, although only to a limited extent, despite its success in some areas. The classic data-driven engineering models frequently narrowly focus on particular system behaviour characteristics or parameters, restricting a whole knowledge of the system.

## Objective:
By investigating the possibilities of using MTL to analyse tabular data obtained from engineering domains, this thesis aims to close this gap. With this method, we want to move beyond the traditional single-parameter concentration and promote a more holistic description of system behaviour. This thesis aims to support the effectiveness and usability of MTL in the engineering domain through a thorough examination of the current literature, the development of synthetic datasets, the use of MTL methodologies, and a baseline comparison to single-task learning.


---

## 📦 Tasks

### Literature Review
- Comprehensive review of existing literature on the generation of synthetic tabular data for MTL.
- Examination of the characteristics defining task relationships within MTL frameworks.
- Analysis of current MTL methods tailored for tabular data.

### Synthetic Data Generation and Analysis
- Development and implementation of methods for generating synthetic tabular data to facilitate a diverse array of scenarios concerning task relationships.
- Comparative analysis of MTL approaches against a single-task learning baseline using generated synthetic data to ascertain the merits and demerits of MTL.

### Real-world Data Validation
- Application and adaptation of the chosen MTL methods on real-world datasets within the domain of process and material engineering.
- Comparative evaluation of MTL and single-task learning on real-world datasets to validate the findings from synthetic data analysis.

### Evaluation and Comparison
- Establishing evaluation metrics to ascertain the performance and applicability of MTL in both synthetic and real-world scenarios.
- Thorough comparison and discussion on the findings to delineate the circumstances under which MTL outperforms single-task learning.

### Conclusion and Future Work
- Summarizing the key findings, challenges encountered, and the implications of this study on the broader engineering domain.
- Suggesting avenues for future research to further the understanding and application of MTL in engineering sciences.

This structured approach aims to provide a well-rounded analysis of MTL's potential in enhancing data-driven modeling in engineering, thereby contributing to a more integrated and comprehensive understanding of complex system behaviors.

---


## 📂 Repository Structure

```sh
└── Multi_gate_Mixture_of_expert_Model_with_synthetic_data_Analysis.git/
    ├── .DS_Store
    ├── MMOE/
    │   ├── __pycache__/
    │   ├── data.py
    │   ├── hyperpar.py
    │   ├── main.py
    │   ├── mmoemodel.py
    │   └── utils.py
    ├── MTLNET_C/
    │   ├── __pycache__/
    │   ├── data.py
    │   ├── hyperpar.py
    │   ├── main.py
    │   ├── mtl_networkC.py
    │   └── utils.py
    ├── MTLpoly/
    │   ├── MTLpoly.py
    │   ├── __pycache__/
    │   ├── data.py
    │   ├── hyperpar.py
    │   ├── main.py
    │   └── utils.py
    ├── MTLwithrealdata/
    │   ├── .DS_Store
    │   ├── Guo_2019/
    │   ├── Hu_2021/
    │   ├── Huang_2021/
    │   ├── Ucb-cbm/
    │   ├── Xiong_2014/
    │   ├── Yin_2021/
    │   └── data/
    ├── README.md
    ├── STL Analysis/
    │   ├── XGBoostmodel.py
    │   ├── Xg.ipynb
    │   ├── __pycache__/
    │   └── utils.py
    ├── data/
    │   ├── .DS_Store
    │   ├── Guo_2019/
    │   ├── Hu_2021/
    │   ├── Huang_2021/
    │   ├── Xiong_2014/
    │   ├── Yin_2021/
    │   └── uci-cbm/
    └── envs/
        └── pytorch.yml
```

---

## 🚀 Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system:

`- ℹ️ python=3.10`

`- ℹ️ torchaudio`

`- ℹ️ torchvision`

`- ℹ️ pytorch`

`- ℹ️ cpuonly1`

`- ℹ️ scikit-learn`

`- ℹ️ plotly`

`- ℹ️ pandas`

`- ℹ️ scikit-optimize`

`- ℹ️ optuna`


### 🔧 Installation

1. Clone the Master_Thesis.git repository:
```sh
git clone https://github.com/QsingularityAi/Master_Thesis.git
```

2. Change to the project directory:
```sh
cd Master_Thesis.git
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🤖 Running Master_Thesis.git

```sh
python main.py
```

### 🧪 Tests
```sh
pytest
```

---


## 🛣 Roadmap

The roadmap below outlines the key milestones and tasks planned for this project to successfully investigate and evaluate the potential of Multi-Task Learning (MTL) in engineering domains.

> - [X] `ℹ️  Task 1: Conduct a comprehensive literature review on synthetic tabular data generation for MTL.`
> - [X] `ℹ️  Task 2: Examine and document the characteristics defining task relationships within MTL frameworks.`
> - [X] `ℹ️  Task 3: Analyze and summarize current MTL methods tailored for tabular data.`
> - [X] `ℹ️  Task 4: Develop and implement methods for generating synthetic tabular data.`
> - [X] `ℹ️  Task 5: Perform a comparative analysis of MTL approaches against a single-task learning baseline using generated synthetic data.`
> - [X] `ℹ️  Task 6: Apply and adapt chosen MTL methods on real-world datasets within the domain of process and material engineering.`
> - [X] `ℹ️  Task 7: Conduct a comparative evaluation of MTL and single-task learning on real-world datasets.`
> - [X] `ℹ️  Task 8: Establish evaluation metrics to gauge the performance and applicability of MTL in both synthetic and real-world scenarios.`
> - [X] `ℹ️  Task 9: Conduct a thorough comparison and discussion on the findings to delineate the circumstances under which MTL outperforms single-task learning.`
> - [X] `ℹ️  Task 10: Summarize key findings, challenges encountered, and the implications of this study on the broader engineering domain.`
> - [X] `ℹ️  Task 11: Suggest avenues for future research to further the understanding and application of MTL in engineering sciences.`

Each task represents a significant step towards achieving the project's objective of enhancing data-driven modeling in engineering through MTL. As tasks are completed, they will be marked accordingly, providing a clear indication of the project's progress.

---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License

This project is licensed under the `ℹ️  LICENSE-TYPE` License. See the [LICENSE-Type](LICENSE) file for additional info.

---

## 👏 Acknowledgments

`- ℹ️ This code bulid by Anurag Trivedi and project offered by faculty of Computer science and Chair of Process informatics and Machine Data Analysis.
feel free reach out to me if you have any Questions on my email aanuragtrivedi007@gmail.com.`

[↑ Return](#Top)

---
