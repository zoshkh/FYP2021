library(GRS.test)
#Change the below working directory to the folder where the GRS files are written using the Python codes
setwd("~path")

data <- read.csv(file = 'GRS_FILE_NAME_1.csv')
data2 <- read.csv(file = 'GRS_FILE_NAME_2.csv')
data3 <- read.csv(file = 'GRS_FILE_NAME_3.csv')

# create factors and return matrices 
# replace n in the below code with the last row number of each dataframe
# replace f1 with the first factor column number
# replace fn with the last factor column number, or remove :fn entirely if the traditional CAPM is tested
factor.mat = data[1:n,f1:fn]
ret.mat = data[1:n, f1:fn]

factor.mat2 = data2[1:n,f1:fn]
ret.mat2 = data2[1:n,f1:fn]

factor.mat3 = data3[1:n,f1:fn]
ret.mat3 = data3[fn]

GRS.test(ret.mat,factor.mat)
GRS.test(ret.mat2,factor.mat2)
GRS.test(ret.mat3,factor.mat3)

