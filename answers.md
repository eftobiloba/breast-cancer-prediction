# Question one
### Change the random state of the Decision tree classifier (for example set it to 42), what was the effect of this change?
The performance improved from 0.91 in accuracy (when the random state was set to 0) to 0.93 in accuracy when the random state was set to 42.

### Performance of the Random Forest Algorithm on the Breast Cancer Dataset

The Random Forest classifier performed **very well** on the breast cancer dataset. Below is a summary of its performance metrics:

| Metric        | Class 0 (Malignant) | Class 1 (Benign) | Overall |
| ------------- | ------------------- | ---------------- | ------- |
| **Precision** | 0.95                | 0.97             | 0.96    |
| **Recall**    | 0.95                | 0.97             | 0.96    |
| **F1-score**  | 0.95                | 0.97             | 0.96    |
| **Accuracy**  | —                   | —                | **96%** |

* **Support**: Class 0 had 63 samples, Class 1 had 108 samples.
* **Macro average** and **weighted average** metrics were both **0.96**, indicating balanced performance across both classes.

✅ **Conclusion**:
The Random Forest algorithm achieved **high accuracy (96%)** with strong precision and recall on both malignant and benign tumor classifications. It is a reliable choice for this dataset.

