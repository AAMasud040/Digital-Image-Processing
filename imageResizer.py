import cv2 as cv

folderlist = ['./Train/colon_aca/colonca','./Train/colon_n/colonn','./Train/lung_aca/lungaca','./Train/lung_n/lungn','./Train/lung_scc/lungscc']
folderlist2 = ['./resized_256_256_3/colon_aca/colonca','./resized_256_256_3/colon_n/colonn','./resized_256_256_3/lung_aca/lungaca','./resized_256_256_3/lung_n/lungn','./resized_256_256_3/lung_scc/lungscc']
for folder in folderlist:
    # folderlist2[folderlist.index(folder)]
    print(folder)
    for i in range(1,5001):
        im = cv.imread(folder+str(i)+'.jpeg') 
        im_r = cv.resize(im,(256,256),interpolation = cv.INTER_AREA)
        cv.imwrite(folderlist2[folderlist.index(folder)]+str(i)+'.jpeg',im_r)

