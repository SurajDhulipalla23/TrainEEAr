# TrainEEAr
Tool Tracking to be used in an Endoscopic Endonasal Approaches Trainer for Neurosurgical residents

## What's the Issue?​
* Endonasal Endoscopic Approaches (EEA’s) are technically challenging and popular minimally invasive approaches to sinonasal and skull base lesions in both Neurosurgery (NSU) and Otolaryngology (ENT)​
* NSU & ENT residents have a lower comfort level in these approaches than in open cases, due to the movements, dangers, and lack of training modalities​

## Components of the TrainEEAr
* Anatomically correct anterior and middle cranial fossa (skull base), and sinus structures​
  * Skull base obtained from a CT scan, which was then rendered into a segmented 3D model using 3D slicer​
  * Superior and posterior skull removed for computer vision and electrical access​
  * PLA 3D printing​
* Computer vision tool tracking to measure tooltip movement and instruct resident how to match expert​
  * Finds max contour in image via masking
  * Kalman filter to smooth tool tracking ​
    * Recursive estimator, noise filter​
* Electronics to detect tool contact with expert-selected “no go zones” (ICAs, Optic Nerves, Cavernous sinuses)​

## Short Youtube Demo
Youtube demo of the Tool Tracking Algorithm (not within the context of the device):
<https://youtube.com/shorts/aZmWDemw04g?feature=shared>
