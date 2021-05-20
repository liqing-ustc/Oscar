export TARGET='https://xiyin1wu2.blob.core.windows.net/maskrcnn/t-lqing/?sv=2020-04-08&se=2021-06-19T20%3A56%3A15Z&sr=c&sp=rwl&sig=Til9uZlFREFVEymAYxmGIA0I1cv%2BYE8%2FAoPIvSPe7bQ%3D'
azcopy cp 'https://biglmdiag.blob.core.windows.net/vinvl/' $TARGET --recursive
