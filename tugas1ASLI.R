library(ggplot2)
library(caret)

N <- 200
D <- 2
K <- 4
X <- data.frame()
y <- data.frame()

set.seed(308)

for (j in (1:K)){
  r <- seq(0.05,1,length.out = N)
  t <- seq((j-1)*4.7,j*4.7, length.out = N) + rnorm(N, sd = 0.3)
  Xtemp <- data.frame(x =r*sin(t) , y = r*cos(t)) 
  ytemp <- data.frame(matrix(j, N, 1))
  X <- rbind(X, Xtemp)
  y <- rbind(y, ytemp)
}

data <- cbind(X,y)
colnames(data) <- c(colnames(X), 'label')

X <- as.matrix(X)
Y <- matrix(0, N*K, K)

for (i in 1:(N*K)){
  Y[i, y[i,]] <- 1
}

nnet <- function(X, Y, step_size = 0.5, reg = 0.001, h = 10, niteration){
  
  N <- nrow(X) 
  K <- ncol(Y) 
  D <- ncol(X) 
  
  
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)
  b <- matrix(0, nrow = 1, ncol = h)
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h)
  b2 <- matrix(0, nrow = 1, ncol = K)
  
  
  for (i in 0:niteration){
    
    hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))
    hidden_layer <- matrix(hidden_layer, nrow = N)
    scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)
    
    
    exp_scores <- exp(scores)
    probs <- exp_scores / rowSums(exp_scores)
    
    corect_logprobs <- -log(probs)
    data_loss <- sum(corect_logprobs*Y)/N
    reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2)
    loss <- data_loss + reg_loss
    
    if (i%%1000 == 0 | i == niteration){
      print(paste("iteration", i,': loss', loss))}
    
    
    dscores <- probs-Y
    dscores <- dscores/N
    
    dW2 <- t(hidden_layer)%*%dscores
    db2 <- colSums(dscores)
    dhidden <- dscores%*%t(W2)
    dhidden[hidden_layer <= 0] <- 0
    dW <- t(X)%*%dhidden
    db <- colSums(dhidden)
    
    dW2 <- dW2 + reg *W2
    dW <- dW + reg *W
    
    W <- W-step_size*dW
    b <- b-step_size*db
    W2 <- W2-step_size*dW2
    b2 <- b2-step_size*db2
  }
  return(list(W, b, W2, b2))
}

nnetPred <- function(X, para = list()){
  W <- para[[1]]
  b <- para[[2]]
  W2 <- para[[3]]
  b2 <- para[[4]]
  
  N <- nrow(X)
  hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T)) 
  hidden_layer <- matrix(hidden_layer, nrow = N)
  scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T) 
  predicted_class <- apply(scores, 1, which.max)

  return(predicted_class)  
}

nnet.model <- nnet(X, Y, step_size = 0.4,reg = 0.0002, h=50, niteration = 50)
predicted_class <- nnetPred(X, nnet.model)
print(paste('training accuracy:',mean(predicted_class == (y))))

hs <- 0.01
x_min <- min(X[,1])-0.2; x_max <- max(X[,1])+0.2
y_min <- min(X[,2])-0.2; y_max <- max(X[,2])+0.2
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = hs), seq(y_min, y_max, by =hs)))
Z <- nnetPred(grid, nnet.model)

ggplot()+
  geom_tile(aes(x = grid[,1],y = grid[,2],fill=as.character(Z)), alpha = 0.3, show.legend = F)+ 
  geom_point(data = data, aes(x=x, y=y, color = as.character(label)), size = 2) +
  xlab('X') + ylab('Y') + ggtitle('Neural Network Decision Boundary') +
  scale_color_discrete(name = 'Label') + coord_fixed(ratio = 0.6) +
  theme(axis.ticks=element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        axis.text=element_blank(),legend.position = 'none')

displayDigit <- function(X){
  m <- matrix(unlist(X),nrow = 28,byrow = T)
  m <- t(apply(m, 2, rev))
  image(m,col=grey.colors(255))
}

train <- read.csv("ml1/train.csv", header = TRUE, stringsAsFactors = F)
displayDigit(train[9,-1])

nzv <- nearZeroVar(train)
nzv.nolabel <- nzv-1

inTrain <- createDataPartition(y=train$label, p=0.7, list=F)

training <- train[inTrain, ]
CV <- train[-inTrain, ]

X <- as.matrix(training[, -1]) 
N <- nrow(X) 
y <- training[, 1] 

K <- length(unique(y)) 
X.proc <- X[, -nzv.nolabel]/max(X) 
D <- ncol(X.proc) 

Xcv <- as.matrix(CV[, -1]) 
ycv <- CV[, 1] 
Xcv.proc <- Xcv[, -nzv.nolabel]/max(X) 

Y <- matrix(0, N, K)

for (i in 1:N){
  Y[i, y[i]+1] <- 1
}

nnet.mnist <- nnet(X.proc, Y, step_size = 0.3, reg = 0.0001, niteration = 50)
predicted_class <- nnetPred(X.proc, nnet.mnist)
print(paste('training set accuracy:',mean(predicted_class == (y+1))))

predicted_class <- nnetPred(Xcv.proc, nnet.mnist)
print(paste('CV accuracy:',mean(predicted_class == (ycv+1))))

Xtest <- Xcv[sample(1:nrow(Xcv), 1), ]
Xtest.proc <- as.matrix(Xtest[-nzv.nolabel], nrow = 1)
predicted_test <- nnetPred(t(Xtest.proc), nnet.mnist)
print(paste('The predicted digit is:',predicted_test-1 ))

displayDigit(Xtest)
