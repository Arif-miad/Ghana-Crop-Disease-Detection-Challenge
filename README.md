# Ghana-Crop-Disease-Detection-Challenge


---

# Ghana Crop Disease Detection Challenge

### Overview
This project is focused on developing a model to detect crop diseases using annotated images. The objective is to identify various diseases present in crops and classify them effectively, contributing to agricultural sustainability and crop health management.

### Dataset
- **License**: [CC BY 4.0 Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- The dataset is divided into:
  - **Training Set**: Each image in the training set includes bounding box annotations for different types of diseases. Multiple diseases may be present within a single image, each annotated separately.
  - **Test Set**: Introduces images that may contain new, unannotated diseases, with no bounding boxes provided. This setup challenges the model's adaptability to novel disease types.
```python
# 1. Distribution of Class
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='class', palette='viridis')
plt.title('Distribution of Crop Disease Classes')
plt.xticks(rotation=45)
plt.savefig("dist_dis.png")
plt.show()
```
![output](https://github.com/Arif-miad/Ghana-Crop-Disease-Detection-Challenge/blob/main/ga1.png)
### Approach
1. **Data Preprocessing**: 
   - Processed images to normalize sizes, apply data augmentation techniques for diversity, and optimize bounding box formats for compatibility with object detection models.
   
2. **Model Selection**:
   - Selected and fine-tuned a deep learning model capable of multi-class object detection to identify and localize diseases using bounding box annotations.
   
3. **Training and Evaluation**:
   - Trained the model on the annotated training set and evaluated its performance using metrics like accuracy, recall, and precision.
   - Focused on generalization to handle unannotated test images with potential new disease types.

4. **Challenges**:
   - Adapting the model to identify unseen diseases in the test set, which required implementing techniques to improve generalization and anomaly detection.

### Results
Achieved a robust model with high accuracy on annotated images and improved adaptability to handle unannotated test data. Contributed to a step forward in disease detection to assist in agriculture and crop management.

### License
This project is released under the **CC BY 4.0 Attribution 4.0 International** license.

---

### Example Usage
```python
# Load and preprocess the dataset
from data_loader import DataLoader
train_data, test_data = DataLoader().load_data()

# Train the model
from model import DiseaseDetectionModel
model = DiseaseDetectionModel()
model.train(train_data)

# Evaluate on test set
results = model.evaluate(test_data)
print("Model Evaluation:", results)
```

### Acknowledgments
Special thanks to the Ghana Crop Disease Detection organizers and the open-source community for the resources and support in this challenge.

