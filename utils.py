tdc_mae_tasks = [ # single task regression
    "tdcommons/lipophilicity_astrazeneca",
    "tdcommons/caco2_wang",
    "tdcommons/ld50_zhu",
    "tdcommons/solubility_aqsoldb",
    "tdcommons/ppbr_az"
]

tdc_auroc_tasks = [ # single task classification
    "tdcommons/bbb_martins",
    "tdcommons/hia_hou",
    "tdcommons/pgp_broccatelli",
    "tdcommons/bioavailability_ma",
    "tdcommons/cyp3a4_substrate_carbonmangels",
    "tdcommons/ames",
    "tdcommons/herg",
    "tdcommons/dili",
]

tdc_spearman_task = [ # single task regression
    "tdcommons/vdss_lombardo",
    "tdcommons/half_life_obach",
    "tdcommons/clearance_microsome_az",
    "tdcommons/clearance_hepatocyte_az",
]

tdc_aucpr_tasks = [ # single task classification
    "tdcommons/cyp2d6_substrate_carbonmangels",
    "tdcommons/cyp2d6_veith",
    "tdcommons/cyp2c9_veith",
]

tdc_aucpr2_tasks = [ # single task classification
    "tdcommons/cyp2c9_substrate_carbonmangels",
    "tdcommons/cyp3a4_veith",
]