# Capstone-Project-Deep-Learning-Analysis-of-Brain-Pathologies
Segmentation of MRI Brain Images with brain tumors and strokes

The purpose of this project is to use deep learning techniques to determine
whether brain pathologies are present in patient magnetic resonance imaging
(MRI) images and to determine which parts of the brain are affected by disease
by localizing the abnormalities, i.e. segmentation of the images. Specifically,
MRI images from patients with brain tumors affecting the glial cells of the brain
which can be classified as high-grade (HGG) with a poor prognosis or low-grade
(LGG) with a better prognosis will be considered. Additionally, patient MRI’s
with brain lesions caused by ischemic stroke will be analyzed. The report 
https://github.com/julie1/Capstone-Project-Deep-Learning-Analysis-of-Brain-Pathologies/blob/master/CapstoneProject.pdf
details my methods and results.  I have used the deepmedic 3-dimensional convolution neural network code https://github.com/Kamnitsask/deepmedic
with several small modifications and some additional code to segment brain MRI images with stroke.
The additional code includes data preparation and a conditional random field.
For segmenting brain MRI's with glial cell tumors (gliomas), the 2-dimensional U-net approach of 
https://github.com/zsdonghao/u-net-brain-tumor proved more successful.  I have used the model and training
code from this repository with modifications, and have added code for data preparation and for distinguishing
low grade gliomas from high grade gliomas. HTML files containing my results
are included in the repository 
