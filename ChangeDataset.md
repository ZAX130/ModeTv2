# How to run the code on other datasets?

1.  Based on the production process of the IXI dataset in TransMorph or the LPBA dataset we provided, create your corresponding .pkl dataset. Or modify the file reading method in **data/datasets.py** according to your needs.
2.  Modify the `self.seg_table` of the class **Seg_norm** in **data/trans.py** to include all the values present in the label images of the dataset, which can typically be obtained using the **np.unique** function.
   
    ![image](https://github.com/ZAX130/ModeTv2/assets/43944700/1a88df49-d6e7-4c4c-8c7f-1cc7447ca560)

    The function of this class is to normalize the values in `seg_table` to a sequence of consecutive integers starting from zero.

4. Modify the `VOI_lbls` in the **dice_val_VOI** function of the **utils.py** to the number of the label (normalized in step 2) that needs to be calculated in Dice metric.
   
   ![image](https://github.com/ZAX130/ModeTv2/assets/43944700/0568e28c-537b-456e-bcfc-c4c7e130ea01)
