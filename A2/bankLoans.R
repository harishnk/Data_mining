# Download RWeka library by typing in the foll. on the console
#install.packages("RWeka", dependencies = TRUE)
#Load the RWeka library as library(RWeka)

bankData = read.csv("/users/harishnk/R/A2/bankData.csv", quote = "\"") 

bankData$dUndergrad <- floor((3-as.numeric(bankData$Education))/2)
bankData$dGrad <- (1 - (as.numeric(bankData$Education) %% 2))

write.csv(bankData, file = "/users/harishnk/R/A2/BankDataModified.csv")


bankDataFrame <- data.frame(bankData)

drops <- c("ID", "ZIP.Code", "Education")

bankDataFrameTrain <- bankDataFrame[1:3000, !(names(bankDataFrame) %in% drops)]

bankDataFrameTest <- bankDataFrame[3001:5000, !(names(bankDataFrame) %in% drops)]

keeps <- c("Personal.Loan")

cl <- bankDataFrame[1:3000, (names(bankDataFrame) %in% keeps)]


knn(bankDataFrameTrain, bankDataFrameTest, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE)

knn.results
#attributes(bankData.knn)

#kNNIBK <- IBk(Personal.Loan ~ ., data = bankDataFrame.df, )

#summary(kNNIBK)






