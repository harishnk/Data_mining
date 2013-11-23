# Download RWeka library by typing in the foll. on the console
#install.packages("RWeka", dependencies = TRUE)
#Load the RWeka library as library(RWeka)

iris <- read.arff(system.file("arff", "iris.arff", package = "RWeka"))

classifier <- IBk(class ~., data = iris)
summary(classifier)