# s2-labeled

Generates a labeled Sentinel-2 scene. The labels are random and use the SCL band to assign a pixel class.

STAC Items are then generated using the label and version STAC extensions.

The labels are used for the AI-Extensions validation using three popular algorithms used for supervised classifications:

* Decision Tree: a non-parametric supervised learning method
* Random Forest: an ensemble classification method, where several predictions of base estimators are combined
* Support Vector Machine (SVM): a set of supervised learning methods used for classification, regression and outliers detection

The example makes use of the algorithms provided by the scikit-learn library.aa