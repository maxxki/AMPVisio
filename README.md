# **AMPVisio**

## **Overview**

AMPVisio is a high-precision, automated optical inspection system designed for the pharmaceutical industry. It utilizes advanced deep learning (YOLOv8) to perform real-time defect detection in ampoules, ensuring 100% inspection reliability for pharmaceutical packaging lines.

## **Key Features**

* **Computer Vision Inspection:** Real-time detection of defects including particles, air bubbles, and cracks using optimized YOLOv8 architectures.  
* **Synthetically Generated Datasets:** Training pipelines leveraging fully synthetic datasets to ensure high variance and perfectly labeled ground truth, reducing the need for massive manual data collection.  
* **GMP Compliance Ready:** Designed with traceability in mind, featuring audit trails (SQLite-based), HMAC-signed log entries, and cryptographic verification of model integrity.  
* **Edge-Optimized:** Engineered for deployment on industrial edge hardware, ensuring high-throughput inspection directly on the production line.  
* **Reproducibility:** Hard-coded seed management and modular architecture for consistent model training and evaluation.

## **Repository Structure**

| Component | Description   |
| :---- | :---- |
| /scripts | Core scripts for dataset preparation and model training. |
| /models | Pre-trained weights and configuration files. |
| /data | Synthetic dataset generation pipelines and schemas. |
| /audit | Logging modules for compliance tracking and system diagnostics. |

## **Technical Specifications**

`- Framework: PyTorch / Ultralytics YOLOv8`  
`- Data Format: Synthetic / Custom Annotation`  
`- Integrity Check: SHA-256 Model Hashing`  
`- Logging: SQLite / Audit-Trail`

## **Compliance**

AMPVisio addresses critical industry requirements for automated visual inspection, including data integrity protocols aligned with FDA 21 CFR Part 11 and EU GMP standards.
