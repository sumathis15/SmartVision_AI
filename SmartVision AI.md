

|  Project Title | SmartVision AI \- Intelligent Multi-Class Object Recognition System |
| :---- | :---- |
| **Skills take away From This Project** | **Python • Deep Learning • TensorFlow/PyTorch • CNN Architectures • Transfer Learning • VGG16 • ResNet50 • MobileNet • EfficientNet • Object Detection • YOLO • Computer Vision • OpenCV • Data Preprocessing • Model Evaluation • Streamlit • Hugging Face • Cloud Deployment • Image Classification • Data Visualization** |
| **Domain** | **Computer Vision & Artificial Intelligence** |

## **Problem Statement**

Object detection and classification are fundamental capabilities in computer vision, powering diverse real-world applications from autonomous vehicles to smart retail, wildlife conservation to traffic monitoring. However, building robust systems that can accurately identify and locate multiple objects across various domains remains a significant challenge.

### **The Challenge:**

Organizations across industries need intelligent systems that can:

* **Accurately detect and classify** objects from diverse categories (vehicles, animals, household items, food, people, etc.)  
* **Handle real-world complexity** with multiple objects present in single images  
* **Provide fast inference** suitable for real-time applications  
* **Demonstrate reliability** across different lighting conditions, angles, and contexts  
* **Scale efficiently** for cloud-based deployment

**The Solution:**

Build a comprehensive computer vision platform that combines:

1. **Transfer Learning-based Image Classification** using state-of-the-art CNN architectures  
2. **YOLO-based Object Detection** for multi-object localization with bounding boxes  
3. **Comparative Analysis** of multiple architectures to identify optimal solutions  
4. **Production-ready Deployment** as an accessible web application

The system should leverage a curated subset of 25 classes from the industry-standard COCO dataset to create a focused yet versatile solution capable of serving multiple business verticals including smart cities, retail analytics, security systems, and automated monitoring

## **Business Use Cases**

### **1\. Smart Cities & Traffic Management**

Automated vehicle detection and counting for traffic flow optimization, pedestrian safety monitoring at crosswalks, parking space management, and traffic violation detection.

### **2\. Retail & E-Commerce**

Product recognition for automated inventory management, customer behavior analytics, scan-free checkout systems, and visual search enabling customers to search products by image.

### **3\. Security & Surveillance**

Intrusion detection identifying unauthorized persons or objects, suspicious object alerts for unattended items, perimeter monitoring for facility access control, and crowd density analysis.

### **4\. Wildlife Conservation**

Automated species identification from camera traps, habitat monitoring tracking animal presence, poaching prevention detecting humans in protected areas, and population studies for conservation research.

### **5\. Healthcare**

Medical equipment tracking in hospitals, PPE compliance verification for healthcare workers, patient fall detection, and hygiene monitoring for proper sanitization protocols.

### **6\. Smart Home & IoT**

Home automation triggering actions based on detected objects, security alerts for unexpected items or people, appliance usage monitoring, and pet activity tracking.

### **7\. Agriculture**

Livestock monitoring and counting, farming equipment location tracking, pest detection in crop areas, and harvest readiness identification for ripe produce.

### **8\. Logistics & Warehousing**

Automated package sorting and routing, real-time inventory tracking, quality control detecting damaged items, and loading bay monitoring for operational efficiency.

## **Approach**

## **Phase 1: Dataset Acquisition & Preprocessing**

## **Step 1.1: Dataset Loading**

The project utilizes a curated subset of the COCO dataset for optimal training efficiency. Given the extensive size of the full COCO dataset (122K+ images), we work with a **focused collection of 2,500 images** (100 images per class across 25 categories), which provides an excellent balance for transfer learning applications.

**Dataset Preparation:**  
 A comprehensive data preparation pipeline has been developed to streamline the dataset loading process. The notebook handles streaming data collection from Hugging Face, automated filtering, preprocessing, and proper structuring for both classification and detection tasks.

📎 **Dataset Preparation Notebook:** [Access Here](https://colab.research.google.com/drive/1KsjW74DSd34IGrqTIqah_jpm_Oyk43Kp?usp=sharing)

**Key Features:**

* Loads COCO dataset from Hugging Face in streaming mode  
* Collects 100 images per class for 25 selected categories  
* Creates organized folder structure for classification and detection  
* Implements train/validation/test splits (70%/15%/15%)  
* Generates cropped objects for classification (224×224 pixels)  
* Prepares YOLO format annotations  
* Produces metadata and configuration files

#### **Step 1.2: Exploratory Data Analysis (EDA)**

Analyze class distribution across selected 25 categories, identify image characteristics and quality, examine objects per image distribution, visualize sample images with annotations, and verify class balance.

#### **Step 1.3: Data Preprocessing for Classification**

Extract bounding boxes from COCO annotations for the 25 selected classes, crop individual objects to create single-object dataset, organize cropped images into 25 class folders, resize images to 224×224 pixels, normalize pixel values, and create train/validation/test splits (70%/15%/15%).

#### **Step 1.4: Data Augmentation**

Apply augmentation techniques including random horizontal flips, rotation (±15 degrees), brightness adjustment (±20%), contrast adjustment, random zoom, and color jittering to increase dataset diversity.

---

### **Phase 2: Transfer Learning \- Image Classification**

#### **Step 2.1: Model 1 \- VGG16**

Load pre-trained VGG16 model with ImageNet weights, freeze convolutional base layers, replace top classification layer for 25 classes, add custom dense layers with dropout, train on cropped single-object images, and save best model weights. Expected performance: \~80-85% accuracy.

#### **Step 2.2: Model 2 \- ResNet50**

Load pre-trained ResNet50 model, implement fine-tuning by unfreezing the last 20 layers, add global average pooling and custom classification head for 25 classes, use learning rate scheduling, implement early stopping, and train on dataset. Expected performance: \~85-90% accuracy.

#### **Step 2.3: Model 3 \- MobileNetV2**

Load pre-trained MobileNetV2 optimized for mobile devices, freeze base and train top layers for 25 classes, add custom classification head, focus on inference speed optimization, and train with standard augmentation. Expected performance: \~82-87% accuracy with faster inference.

#### **Step 2.4: Model 4 \- EfficientNetB0**

Load pre-trained EfficientNetB0 with compound scaling, implement fine-tuning with mixed precision training for 25 classes, add classification head with batch normalization, train with advanced augmentation, and optimize for best accuracy. Expected performance: \~88-93% accuracy.

#### **Step 2.5: Model Comparison & Selection**

Compare all 4 models on the test set using accuracy, precision, recall, F1-score, inference time, and model size metrics. Generate confusion matrices, create comparison visualizations, identify top-5 accuracy for each model, select the best model based on accuracy-speed tradeoff, and document findings.

---

### **Phase 3: Object Detection with YOLO**

#### **Step 3.1: YOLOv8 Setup**

Install Ultralytics YOLOv8 library, understand YOLO architecture and detection pipeline, download pre-trained YOLOv8 weights trained on COCO, and configure models for the 25 selected classes.

#### **Step 3.2: Dataset Preparation for Detection**

Filter COCO dataset for the 25 selected classes, convert annotations to YOLO format if needed, prepare dataset structure with organized images and labels folders, create data.yaml configuration file, and verify bounding box annotations.

#### **Step 3.3: YOLO Training/Fine-tuning**

Use pre-trained YOLOv8 and fine-tune on the 25-class subset. Configure training parameters including epochs, batch size, image size, and optimizer. Monitor training metrics including mAP, precision, and recall. Save best performing weights. Expected performance: mAP@0.5 \> 85%, inference speed: 30-50 FPS.

#### **Step 3.4: YOLO Evaluation**

Test on validation set, calculate detection metrics (mAP@0.5, mAP@0.5:0.95, precision, recall per class, inference speed), visualize predictions on sample images, and analyze failure cases and limitations.

---

### **Phase 4: Model Integration & Pipeline Development**

#### **Step 4.1: Inference Pipeline Design**

Create end-to-end prediction pipeline: user uploads image → YOLO detects all objects with bounding boxes → optional classification verification using best CNN model → display image with bounding boxes, class labels, and confidence scores.

#### **Step 4.2: Post-processing**

Apply Non-Maximum Suppression (NMS) to remove duplicate detections, filter predictions using confidence threshold (\>50%), refine and visualize bounding boxes, and format class labels.

#### **Step 4.3: Performance Optimization**

Implement model quantization for faster inference, enable batch processing for multiple images, configure GPU acceleration, and optimize memory for cloud deployment.

### **Phase 5: Streamlit Application Development**

#### **Step 5.1: Application Architecture**

Design multi-page Streamlit application with the following pages:

**Page 1: Home** \- Project overview, key features, instructions, and sample demo images

**Page 2: Image Classification** \- Upload image interface, single object classification using trained CNNs, display predictions from all 4 models, show top-5 predictions with confidence scores, model comparison side-by-side

**Page 3: Object Detection** \- Upload image interface, YOLO detection with bounding boxes, display all detected objects with labels, show detection confidence scores, option to adjust confidence threshold

**Page 4: Model Performance** \- Model comparison dashboard, accuracy metrics visualization, inference speed comparison, confusion matrices, class-wise performance breakdown

**Page 5: Live Webcam Detection (Optional)** \- Real-time webcam feed, live object detection using YOLO, display FPS and latency metrics

**Page 6: About** \- Project documentation, dataset information, model architectures used, technical stack details, developer information

---

### **Phase 6: Deployment on Hugging Face Spaces**

#### **Step 6.1: Deployment Preparation**

Create a GitHub repository with project code, prepare requirements.txt with all dependencies, create README.md with setup instructions, optimize model files if needed, and use Git LFS for large model files (\>100MB).

#### **Step 6.2: Hugging Face Spaces Configuration**

Create new Space on Hugging Face, select Streamlit as SDK, connect GitHub repository, configure Space settings including Python version and hardware, add secrets or environment variables if needed.

#### **Step 6.3: Testing & Optimization**

Test all features in the production environment, monitor memory usage and optimize if needed, ensure models load correctly from cloud storage, test with various image types and sizes, verify cross-browser compatibility, and check mobile responsiveness.

#### **Step 6.4: Documentation**

Create comprehensive README with project description, usage instructions, model performance metrics, dataset information, and installation instructions for local setup.

**Results** 

### **Expected Technical Outcomes:**

**Classification Models Performance:**

* VGG16: 80-85% accuracy, \~150ms inference  
* ResNet50: 85-90% accuracy, \~100ms inference  
* MobileNetV2: 82-87% accuracy, \~50ms inference  
* EfficientNetB0: 88-93% accuracy, \~80ms inference

**Object Detection Performance:**

* YOLOv8 mAP@0.5: 85-90%  
* Inference Speed: 30-50 FPS on GPU  
* Multi-object detection: 1-10+ objects per image  
* Processing time: \<2 seconds per image

**System Performance:**

* Successfully deployed web application on Hugging Face Spaces  
* Real-time inference with visual feedback  
* Responsive UI accessible on desktop and mobile devices

**Business Impact:**

**Operational Efficiency:** 70% reduction in manual image annotation time, 24/7 automated monitoring capability, real-time processing enabling immediate decision-making, scalable solution handling thousands of images per hour.

**Cost Savings:** 60% lower infrastructure costs compared to specialized hardware solutions, cloud-based deployment eliminating on-premise servers, open-source models reducing licensing fees, automated workflows reducing human labor requirements.

**Market Applications:** Applicable across 8+ industries including retail, security, traffic, wildlife, healthcare, smart homes, agriculture, and logistics. Enables new business models through visual search, automated quality control, and smart analytics.

## **Project Evaluation Metrics**

### **Total Score: 100 Points**

**1\. Data Preprocessing & EDA (15 points)**

* Dataset loading and EDA with visualizations (8 points)  
* Object extraction and data splitting (7 points)

**2\. Transfer Learning Models \- Classification (30 points)**

* Implementation of 4 models: VGG16, ResNet50, MobileNet, EfficientNet on 25 classes (20 points)  
* Minimum 80% accuracy on test set with proper training pipeline (10 points)

**3\. Object Detection \- YOLO (25 points)**

* YOLOv8 implementation and dataset preparation for 25 classes (15 points)  
* Multi-object detection with mAP@0.5 \> 75% (10 points)

**4\. Model Comparison & Analysis (10 points)**

* Comprehensive comparison of all models with performance metrics and visualizations (10 points)

**5\. Streamlit Application Development (15 points)**

* Multi-page functional application with classification and detection features (10 points)  
* Clean UI/UX with proper error handling (5 points)

**6\. Deployment on Hugging Face Spaces (5 points)**

* Successful deployment with all features working in production (5 points)

**Bonus Points (Up to \+10):**

Real-time webcam detection (+3), Advanced augmentation (+2), Model ensemble (+3), Comprehensive documentation (+2).

## **Technical Tags**

Python • Deep Learning • CNN • VGG16 • ResNet50 • MobileNet • EfficientNet • Transfer Learning • YOLO • YOLOv8 • Object Detection • Image Classification • Computer Vision • OpenCV • COCO Dataset • Hugging Face • TensorFlow • PyTorch • Streamlit • Cloud Deployment • Model Evaluation • Data Augmentation • Bounding Boxes • mAP • Precision • Recall.

## **Data Set**

### **Dataset Name: COCO (Common Objects in Context) 2017 \- 25 Class Subset**

### **Source:**

* **Hugging Face Repository**: `detection-datasets/coco`  
* **Official Website**: [https://cocodataset.org/](https://cocodataset.org/)  
* **Direct Hugging Face Link**: [Coco Dataset](https://huggingface.co/datasets/detection-datasets/coco)  
* **🔗 Dataset Preparation Colab:** [Smartvision.ipynb](https://colab.research.google.com/drive/1KsjW74DSd34IGrqTIqah_jpm_Oyk43Kp?usp=sharing)

### **Project Dataset Scope:**

* **Selected Classes**: 25 most common and diverse object categories  
* **Total Images**: Filtered subset from 122,218 COCO images  
* **Image Format**: JPEG (RGB color images)  
* **Image Sizes**: Variable (typical range: 640×480 to 1920×1080 pixels)  
* **Annotation Format**: COCO JSON format with bounding boxes

### **25 Selected Object Classes:**

The project focuses on 25 carefully selected classes that provide good representation across different domains while maintaining computational feasibility:

| Category | Classes | Count |
| :---- | :---- | :---- |
| 🚗 **VEHICLES** | car, truck, bus, motorcycle, bicycle, airplane | 6 |
| 👤 **PERSON** | person | 1 |
| 🚦 **OUTDOOR** | traffic light, stop sign, bench | 3 |
| 🐾 **ANIMALS** | dog, cat, horse, bird, cow, elephant | 6 |
| 🍽️ **KITCHEN & FOOD** | bottle, cup, bowl, pizza, cake | 5 |
| 🪑 **FURNITURE & INDOOR** | chair, couch, bed, potted plant | 4 |

**Total: 25 Classes**

**Why These 25 Classes?**

* ✅ Most frequently occurring in COCO dataset (good data availability)  
* ✅ Visually distinct categories (easier to classify accurately)  
* ✅ Represent diverse real-world scenarios (vehicles, animals, indoor, outdoor)  
* ✅ Balanced complexity (mix of easy and challenging objects)  
* ✅ Practical business applications across multiple industries

### 

### 

### 

### 

### 

### **Dataset Structure:**

![][image1]

### **Annotation Format (JSON):**

![][image2]

**How to Load Dataset:**

The dataset is loaded using a streaming approach from Hugging Face, which efficiently handles the large COCO dataset without requiring full download. The provided Colab notebook implements this streaming methodology to filter and collect exactly 100 images per class across the 25 selected categories.

**Using the Dataset Preparation Notebook:**

The notebook streams the COCO dataset and applies filtering logic to collect 2,500 images total:

**📎 Access the complete implementation:** [Dataset Preparation Colab](https://colab.research.google.com/drive/1KsjW74DSd34IGrqTIqah_jpm_Oyk43Kp?usp=sharing)

The notebook handles:

* Streaming data collection from Hugging Face (no full download required)  
* Filtering for 25 target classes  
* Collecting exactly 100 images per class (2,500 images total)  
* Organizing images into proper folder structure  
* Creating train/val/test splits  
* Generating YOLO format annotations

## **Data Set Explanation**

### **Dataset Overview:**

### The COCO (Common Objects in Context) dataset is one of the most widely-used benchmarks in computer vision for object detection, segmentation, and captioning tasks. Released by Microsoft in 2014, COCO has become the industry standard for evaluating object detection models.

**Project Dataset Approach:**

This project leverages a carefully curated subset of 2,500 images (100 per class) from the full COCO dataset. This focused approach is particularly well-suited for transfer learning applications, where pre-trained models already possess robust feature extraction capabilities from training on millions of images. The balanced class distribution ensures fair model evaluation and efficient training cycles suitable for both educational purposes and production deployment.

**Dataset Preparation:**  
 A streamlined data preparation pipeline is available via Google Colab, handling all aspects of data collection, preprocessing, and organization. This ensures consistency and reproducibility across the project workflow.

📎 **Reference:** [Dataset Preparation Notebook](https://colab.research.google.com/drive/1KsjW74DSd34IGrqTIqah_jpm_Oyk43Kp?usp=sharing)

### **Key Characteristics:**

* **Real-world Images:** Contains everyday scenes with complex backgrounds  
* **Multi-object Images:** Most images contain multiple objects (average 7.3 objects per image)  
* **Diverse Contexts:** Objects appear in various poses, scales, and occlusions  
* **Rich Annotations:** Precise bounding boxes with class labels  
* **Balanced Classes:** Our 25-class subset maintains equal distribution (100 images each)

### **Input Features:**

#### **1\. Images:**

* **Format**: RGB color images in JPEG format  
* **Resolution**: Variable sizes (typically 640×480 to 1920×1080)  
* **Content**: Everyday scenes including indoor/outdoor, urban/rural, daytime/nighttime  
* **Quality**: High-quality photographs from diverse sources  
* **Preprocessing Required**:  
  * Resize to standard dimensions (e.g., 224×224 for classification, 640×640 for YOLO)  
  * Normalize pixel values to \[0, 1\] or standardize using ImageNet mean/std  
  * Convert to appropriate tensor format for deep learning frameworks

#### **2\. Bounding Boxes:**

* **Format**: \[x, y, width, height\] where:  
  * `x, y`: Top-left corner coordinates  
  * `width, height`: Box dimensions in pixels  
* **Precision**: Pixel-level accuracy  
* **Multiple Boxes**: Each image can have 1 to 50+ bounding boxes  
* **Annotations Include**:  
  * Object category ID (1-90, mapped to 80 classes)  
  * Segmentation masks (for advanced tasks, optional for this project)  
  * Area of the object  
  * Crowd flag (indicating whether annotation is for a single object or crowd)

#### **3\. Category Labels:**

* **25 Curated Categories:** Ranging from frequently occurring objects (person, car, chair) to specialized items (traffic light, potted plant, airplane)  
* **Class IDs**: Integers from 1 to 90 (some IDs skipped, 80 active classes)  
* **Hierarchical Structure**: Classes organized into super-categories (person, vehicle, animal, etc.)

### **Target Variables:**

#### **For Classification Task:**

* **Single Label per Cropped Image**: After extracting objects using bounding boxes  
* **Output Format**: One-hot encoded vector of size 25  
* **Example**: Image of a dog → \[0, 0, ..., 1, ..., 0\] where 1 is at index for "dog" class

#### **For Object Detection Task:**

* **Multiple Outputs per Image**:  
  * Bounding box coordinates: \[x, y, w, h\]  
  * Class label: Integer (0-24)  
  * Confidence score: Float (0.0-1.0)  
* **Variable Number**: Each image can have different number of detections

**Example Output**:  
 ![][image3]

### **Data Distribution:**

#### **Class Frequency (Top 10 Most Common):**

Each class has **exactly 100 images**, ensuring:

* ✅ Perfect class balance  
* ✅ No class dominates training  
* ✅ Fair evaluation across all categories  
* ✅ Equal learning opportunity for each class

**Implication:** Unlike the full COCO dataset where "person" dominates with 40% of annotations, our subset provides equal representation, leading to more balanced model performance.

## **Project Deliverables**

### **1\. Source Code & Notebooks**

* **Dataset Preparation Notebook** \- Automated pipeline for data collection and preprocessing  
   📎 [Colab Link](https://colab.research.google.com/drive/1KsjW74DSd34IGrqTIqah_jpm_Oyk43Kp?usp=sharing)  
* Jupyter notebooks for EDA, preprocessing, model training (VGG16, ResNet50, MobileNet, EfficientNet), YOLO detection, and pipeline integration  
* Python scripts including main Streamlit app, utility functions, and configuration files  
* Requirements.txt and README.md documentation

### **2\. Trained Models**

Four classification models (VGG16, ResNet50, MobileNetV2, EfficientNetB0) and one YOLO detection model with saved weights. Model performance metrics in JSON format.

### **3\. Documentation**

Comprehensive README with project overview, installation instructions, usage guide, model descriptions, and performance metrics. Technical report covering methodology, EDA findings, model training process, and deployment details.

### **4\. Streamlit Web Application**

Deployed application on Hugging Face Spaces with public URL. Multi-page interface including Home, Classification, Detection, Model Performance, and About pages. Features include image upload, predictions from all models, bounding box visualization, and performance dashboards.

## **Project Guidelines**

### **Best Practices**

Write clean, well-documented code following PEP 8 standards. Use meaningful variable and function names with proper docstrings. Initialize Git repository from the beginning with frequent, meaningful commits. Document hyperparameters and track model versions. Save best model checkpoints during training. Test components individually before integration and validate on diverse test images.

## **Timeline**

The project should be completed and submitted within **14 days** from the date it is assigned.

**References: ** 

| TOPIC | LINK |
| :---- | :---- |
| **Dataset Preparation Colab** | [Smartvision.ipynb](https://colab.research.google.com/drive/1KsjW74DSd34IGrqTIqah_jpm_Oyk43Kp?usp=sharing)  |
| **Project Live Evaluation** | [Project Live Evaluation](https://docs.google.com/document/u/0/d/1QisLD2kqDWFZJG2oDknKn2eMGi-Xq8oFPgA7UWSbcIQ/edit) |
| **EDA Guide** | [Exploratory Data Analysis (EDA) Guide](https://docs.google.com/document/d/1tHiTU1X9UwXSLySpJ-FVCohlf_8xpXwa75vlK9S6wl8/edit?usp=sharing) |
| **Capstone Explanation Guideline** | [Capstone Explanation Guideline](https://docs.google.com/document/d/1gbhLvJYY7J73lu1g9c6C9LRJvYemiDOdRDAEMe632w8/edit) |
| **GitHub Reference** | [How to Use GitHub.pptx](https://docs.google.com/presentation/d/1XHCbgUOqbcXNUyQ87vTlKdKRgAbBxtkA/edit?usp=sharing&ouid=109735616107417446342&rtpof=true&sd=true) |
| **Project Orientation (English)** |  |
| **Project Orientation (Tamil)** | [Smart vision Ai orientation - Tamil](https://drive.google.com/file/d/1ml0ejUrPuhRGsIUBcgb5y_olgFfZsaDF/view?usp=sharing) |
| **STREAMLIT RECORDING (Tamil)** | [Streamlit - Tamil](https://drive.google.com/file/d/1NEQRqhHHIWT1G04WN5-DxlGcLx2kjlDI/view?usp=sharing) |
| **STREAMLIT RECORDING (English)** | [Special session for STREAMLIT(11/08/2024)](https://docs.google.com/document/d/1aR3pUZFlCi8gicpF6aPHPESeFdOtGMlfob5PckresZk/edit?usp=sharing) |
| **STREAMLIT DOCUMENTATION** | [Install Streamlit](https://docs.streamlit.io/get-started/installation) |
| **Flask Reference Docs** | [Introduction to Flask (2).pdf](https://drive.google.com/file/d/1LlkSoYQRRBVqcnqJrqGTTp957X3CpRD9/view?usp=sharing) |
| **Deep Learning tutorial** | **[Deep Learning](https://docs.google.com/spreadsheets/d/1CbwUp56oGQ99BX8Y_8rjiiBJGrTu6P3SvI615zxCjD8/edit?gid=0#gid=0)** |
| **Project Excellence Series \[Deep Learning\] (English)** | **[Project Excellence Series: Guided Learning & Problem Solving \[Deep Learning\](English)](https://docs.google.com/document/d/1Ddo8-ik1Gn_RhVpLASga5eX7uqjj_NgmmPw0zVPw1B0/edit?usp=sharing)** |
| **Project Excellence Series \[Deep Learning\] (Tamil)** | **[Project Excellence Series: Guided Learning & Problem Solving \[Deep Learning\](Tamil)](https://docs.google.com/document/d/1Yx-Hc6sKktVf0VZSnCaC_zgriNl1XBBIF0WKSO8Pss4/edit?usp=sharing)** |
| **Hugging Face Spaces Guide** | [**Hugging face spaces**](https://huggingface.co/docs/hub/spaces) |
| **VGG16** | [**VGG Documentation**](https://arxiv.org/abs/1409.1556) |
| **ResNet50** | [**Resnet Documentation**](https://arxiv.org/abs/1512.03385) |
| **MobileNetV2** | **[MobileNetV2 Documentation](https://arxiv.org/abs/1801.04381)** |
| **EfficientNet** | **[EfficientNet Documentation](https://arxiv.org/abs/1905.11946)** |
| **YOLO** | **[YOLO Documentation](https://docs.ultralytics.com/)** |

**PROJECT DOUBT CLARIFICATION SESSION ( PROJECT AND CLASS DOUBTS)**

**About Session:** The Project Doubt Clarification Session is a helpful resource for resolving questions and concerns about projects and class topics. It provides support in understanding project requirements, addressing code issues, and clarifying class concepts. The session aims to enhance comprehension and provide guidance to overcome challenges effectively.  
**Note: Book the slot at least before 12:00 Pm on the same day**

**Timing: Monday-Saturday (3:30PM to 4:30PM)**

**Link to Join: [Project Doubt Session \[DS/AIML\]](https://docs.google.com/document/d/1J7z_Y2whgCtFDXixysXBNXqCuhXAYD9vkiNQ-7Q7XKs/edit?usp=sharing)**

**LIVE EVALUATION SESSION (CAPSTONE AND FINAL PROJECT)**

**About Session:** The Live Evaluation Session for Capstone and Final Projects allows participants to showcase their projects and receive real-time feedback for improvement. It assesses project quality and provides an opportunity for discussion and evaluation.  
**Note: This form will Open only on Saturday (after 2 PM ) and Sunday on Every Week**

**Timing:**   
**For DS and AIML**  
**Monday-Saturday (05:30PM to 07:00PM)**

**Booking link : [https://forms.gle/1m2Gsro41fLtZurR](https://forms.gle/1m2Gsro41fLtZurRA)**

    

| Created By: | Verified By: | Approved By: |
| :---- | :---- | :---- |
| **Subhash Govindharaj** | **[Nehlath Harmain](mailto:nehlath@guvi.in)** | **[Nehlath Harmain](mailto:nehlath@guvi.in)** |

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmcAAAFZCAYAAADdKohRAAA+nElEQVR4Xu2d+7cU1Z235/+YaBQDCCpXBREFAbkoKoIIKIiCoNwvXhCUm8hViIIak8w7Zsw7b2byzpqs3GatmSRmsmZW3plkTeaSH0xijGPGJBoZjaNCVFD7Pd+SXdn9qV3dfc6p3ef0Ps+z1rOqateuXdV9cPfH6uq9/6gGEIlP/PEfIyJiRCFN/kgLAKpCOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDOxFERKxWSBPCGURDO5H+7OxrrimUlXn9ddcVyqpyyODBhTJExDIhTQhnEA3tRNrp6dOnC2VlfvLss2sffvhhobzMmK/tysmTC2UhX3nllUJZFTZ6becPHVq7etasQjki9p2QJoQziIZ2Iup5gwbVJlx6aaH8rE98Ilt+6rzz6srHXXJJoa7z0vHj67a7E87KvPLKKwtlZU687LJCmTlu3LhCWcizzzqrNmbMmEI4G9tVdsHw4YX6oXBm79uoUaMK5aO7ys755CcL5ZdPnFgoa/R3a7QPEftGSBPCGURDOxHfn/zkJ3kIcwHtt6++Wlt11135sW75zW98o64jGnTuuXXbFvJs/St//dd5uYYz/3r8dVPvnJ1zzjm1zZs3Z+vuGsvaGj5sWL79wQcf1H70L/+S13n77bcL9UPeNH9+bfOmTYW6J0+ezNe1DQ1n/vX7df31gwcOZMufPPdcdq3uuDtXrgzW9505Y0bhPUXEvhfShHAG0dBOxPdnP/tZbd3atXVlFs6GnX9+fuwbb7yR7xszenTt2Wefzcr+15/8SV3HpG2bGiQMd/dIj9FwZtu/+tWvCm36bfnrCxcsKOzTOtpGWXtr5T05duxY9rq1DQ1npt1ZtLofffRRXduXXHxxXT3ja1/7Wu0bX/967W/+5m/q2tbzNCtHxL4V0oRwBtHQTkT953/+57q6Fs7sq0y3ffz48WxpwennP/95tv7drvDxpS99qa5j0nZNDWf2oL0LYDt2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RBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTQhnEA3tRBARsVohTf4/Kr10RXjjBJAAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>