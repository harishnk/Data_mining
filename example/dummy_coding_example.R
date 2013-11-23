#payData = read.delim("/users/harishnk/R/example/#dataset_multipleRegression_categorical.csv", header=TRUE, #sep="\t", quote = "\"")

#names(payData)
#attach(payData)

#print(CONF)

payData = read.csv("/users/harishnk/R/example/dataset_multipleRegression_categorical.csv")

payData$dCONF <- as.numeric(payData$CONF)-1

print(payData)

#detach(payData)


