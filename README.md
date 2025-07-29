### 👩‍💻 Authors

- [Samyuktha Nair](https://www.linkedin.com/in/samyuktha-nair-b8284529a/)
- [Arnav Nair](https://www.linkedin.com/in/arnav-nair-2b1774377/)
- [Advay Dinesh](https://www.linkedin.com/in/advay-dinesh-6741aa30a/)
- [Advait Baijulal](https://www.linkedin.com/in/advait-baijulal-a61b892a6/)

### 🧑‍🏫 Mentor and Project Lead

- [Madhuvanthi Venkatesh](https://www.linkedin.com/in/madhuvanthi-venkatesh-a1836192/)


## Text classification using BERT, RoBERTa, DistilBERT and TinyBERT.
This repository presents a technical study on We evaluate and compare four encoder-based architectures — BERT, RoBERTa, DistilBERT, and TinyBERT — all trained on a shared dataset. The models are assessed on classification accuracy, efficiency, and inference speed to explore the trade-offs between size and performance. 

## Dataset Description

<img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/18bee7d9-837e-41f3-9ea2-c6ac5fec1b37" />

The dataset used for differentiating between human-written and AI-generated text consists of a list of strings with an equal representation of both types. Equal representation is crucial as it ensures balanced learning for the model, reducing bias during training.

Each entry in the dataset is labeled with a value of `0` (human-written) or `1` (AI-generated). This binary labeling is necessary for binary classification tasks and ensures the model is trained fairly across both categories.

Working with equal proportions of both types minimizes skewed predictions and enables more accurate evaluation of true model performance. The dataset contains **20,283 entries**, providing a substantial number of examples for both training and evaluation. Datasets used in this project were sourced from Kaggle.

**Source 1** https://www.kaggle.com/code/nirmalgaud/human-vs-ai-text

**Source 2** https://www.kaggle.com/datasets/athena21/ai-ga-dataset


## Overview of BERT and Its Variants

**BERT** (Bidirectional Encoder Representations from Transformers) is an encoder-only transformer architecture developed in 2018 by researchers at **Google AI Language**. It introduced a major shift in natural language processing by leveraging **deep bidirectional attention**, which allows the model to consider the full context of a word based on both its left and right surroundings. BERT is pre-trained on large corpora like **Wikipedia** and **BooksCorpus** using two objectives: **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**. These pretraining tasks help BERT learn rich contextual relationships between words and sentences, improving performance on various downstream tasks like **question answering**, **sentence classification**, and **named entity recognition**.

While BERT was a major milestone in NLP, further enhancements led to more optimized variants. **RoBERTa** (Robustly Optimized BERT Approach), introduced by **Facebook AI**, builds on BERT by **removing the NSP task**, **training on more data** for longer periods, using **larger batch sizes**, and applying **dynamic masking**. These changes significantly improve performance, though RoBERTa is more computationally demanding.

**DistilBERT**, developed by **Hugging Face**, is a smaller, faster version of BERT created using a technique called **knowledge distillation**, where a student model learns to mimic the outputs of a larger teacher model. DistilBERT retains about **97% of BERT’s performance** while being **40% smaller** and **60% faster**, making it suitable for **real-time applications and APIs**.

**TinyBERT** is the most compact variant among the four. It also uses knowledge distillation but includes both **general-purpose and task-specific distillation** in a **two-stage training process**. TinyBERT benefits from **layer-wise distillation** and is trained using both **BERT-Base (12 layers)** and **BERT-Large (24 layers)** as teacher models. It has **fewer parameters** than DistilBERT—around **14.5 million** in its 4-layer version—yet delivers **strong performance**, particularly when optimized for specific tasks.


## Expected Performance of BERT and Variants

The table below summarizes the key characteristics and performance trade-offs of several popular BERT-based models used in this study:

| **Model**        | **Key Innovation**                                                        | **Approx. Parameters** | **Relative Size** | **Relative Speed** | **Core Use Case**                             |
|------------------|---------------------------------------------------------------------------|------------------------|-------------------|--------------------|-----------------------------------------------|
| **BERT-Base**    | Deep Bidirectionality (MLM + NSP)                                         | 110M                   | 1x                | 1x                 | General-purpose NLP, fine-tuning              |
| **RoBERTa-Base** | Optimized Pre-training (No NSP, More Data, Dynamic Masking)               | 125M                   | ~1.1x             | ~1x                | Research, state-of-the-art NLP tasks          |
| **DistilBERT**   | Knowledge Distillation (Triple Loss)                                      | 66M                    | ~0.6x             | ~1.6x faster       | Production systems, real-time APIs            |
| **TinyBERT (4L)**| Two-Stage Layer-wise Distillation (from BERT-Base and BERT-Large)         | 14.5M                  | ~0.13x            | ~9.4x faster       | Edge devices, mobile applications             |

---

## Key Results & Insights

- **Longer input texts improved model prediction accuracy.** All BERT models performed better on long paragraphs than short texts.
- **Human-written text was easier to classify** correctly than AI-generated text across all models.
- **TinyBERT and DistilBERT offered the best balance** of speed and accuracy.

### Confusion Matrix Summary
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/16320d0d-35a5-4a7e-98a0-61c86c9ebe2f" />

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/730714b0-f5f6-4fee-8dfa-84498975b9c1" />


- **BERT**: Strong accuracy, minimal false positives/negatives.
- **RoBERTa**: High recall, but major drop in precision due to frequent misclassification of human text as AI.
- **DistilBERT**: Comparable accuracy to BERT with faster training.
- **TinyBERT**: Slightly lower precision, but excellent speed and efficiency.

---

## Performance Comparison

| Model        | Avg. Training Time | Epochs | Accuracy | Precision | Recall | F1-Score |
|--------------|--------------------|--------|----------|-----------|--------|----------|
| **BERT**     | 01:22:17           | 10     | 0.9931   | 0.9911    | 0.9939 | 0.9968   |
| **RoBERTa**  | 00:39:04           | 10     | 0.6516   | 0.5032    | 0.6694 | 0.9994   |
| **DistilBERT** | 00:39:07         | 10     | 0.9948   | 0.9894    | 0.9942 | 0.9990   |
| **TinyBERT** | 00:03:10           | 5      | 0.9876   | 0.9802    | 0.9871 | 0.9940   |

---

## Training Loss Insights

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/f9cd6401-579d-487e-a22e-9273337ef478" />


- **BERT** and **TinyBERT** had smooth, consistent reductions in training loss.
- **DistilBERT** showed small fluctuations after epoch 4 due to possible overshooting from a higher learning rate.
- **RoBERTa** experienced a sharp spike in loss around epoch 9, likely due to overfitting or batch size imbalance.

---

## Evaluation Metrics

All models were evaluated using the following standard metrics:

- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of positive predictions that were correct.
- **Recall**: Proportion of actual positives that were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.

> All models were trained on Google Colab Pro using a Tesla T4 GPU.

---

## Conclusion

This study shows that transformer-based models can effectively classify AI vs. human-written text, particularly when input length is sufficient. Key takeaways:

- **DistilBERT** offered the best overall performance and efficiency.
- **TinyBERT** is ideal for real-time or resource-limited deployments.
- **BERT** remains a strong baseline with excellent accuracy.
- **RoBERTa**, while promising in theory, underperformed in this task due to overprediction of AI class.

**For full methodology, model architecture, experiments, and additional visuals (e.g., training loss graphs and confusion matrices), refer to the full paper included in this repository.**


## Contributing

We welcome contributions of all kinds — bug fixes, model improvements, dataset enhancements, or documentation updates.

## Using This Project for Teaching or Training

You are welcome to use this project for **non-commercial educational and research purposes**, such as:

- Machine learning and NLP courses  
- Lectures or tutorials  
- Workshops or academic research  
- Demos related to transformer-based classification
  

## Contact

For any questions, feedback, collaboration opportunities, or permissions, feel free to get in touch:

- **Email:** madhuvanthivenkatesh@gmail.com  
- **LinkedIn:** [https://www.linkedin.com/in/madhuvanthi-venkatesh-a1836192/
](https://www.linkedin.com/in/madhuvanthi-venkatesh-a1836192/)
