## Paper Mapping the Future: Revealing Habitat Preferences and Pat-terns of the 
##dangered Chilean Dolphin in Seno Skyring, Patagonia

#### PERFORMING DIFFERENT METHODS TO PREDICT PRESENCE/ABSENCE DAUPHIN CHILIEN-
####                            SKYRING

# Liliana Perez, Yenny Cuellar, Jorge Gibbons, Elias Pinilla Matamala, 
# Simon Demers, Juan Capella

#setting libraries
library(raster)
library(rgdal)
library(sf)
install.packages("rsample")##for splitting observations data
library(rsample)
library(randomForest) ## Random Forestm method
library(glue)## to Generalized Linear Model (GLM)
install.packages("Rtools")
install.packages("caret")
install.packages("recipes")
library(recipes)
library(hardhat)
library(caret)##  TO GLM
library(maxnet)  ## TO Maxent algorithm
library(glmnet)
library(dismo)
library(tibble)
install.packages("nnet")
library(nnet)
#####
#reading data
bathymetry=raster("bathymetry.tif")
fish_farming=raster("fish_farming.tif")
kelp=raster("kelp.tif")
oxygen_dissolved=raster("oxygen_dissolved.tif")
salinity=raster("salinity.tif")
silice=raster("silice.tif")
temp_seafloor=raster("temperature_seafloor.tif")
obs_csv=read.csv("observations.csv", sep = ";")
## shp study area
pol=read_sf("polygonSkyring.shp")
coordinates(obs_csv)= ~ x + y ## TO SET x and y as coordinates
#Assembling all predictor variable values into one RasterBrick
data=stack(bathymetry,fish_farming,kelp,oxygen_dissolved,salinity,silice,temp_seafloor)
#extract raster value by points
rasvalue=extract(data,obs_csv)
combinePointValue=cbind(obs_csv,rasvalue)
all_data=as.data.frame(combinePointValue)
drops <- c("?..FID","Shape..","POINTID","y","x") #taking fields to delete
all_data2=all_data[ , !(names(all_data) %in% drops)] #to obtain just the fields of interest

########### Create training (70%) and test (30%) dataset ##############
set.seed(3)
data_split=initial_split(all_data2,prop = 0.7)
train=training(data_split) ## 70%
test=testing(data_split)  ## 30%

presence=subset(all_data2,Presence==1)
absence=subset(all_data2,Presence==0)

#set number of folds to use

folds=5

#partiction presence and absence data according to folds using the kfold() function.

kfold_pres <- kfold(presence, folds)
kfold_back <- kfold(absence, folds)

######################################## 1 - GLM ###############################

#create an empty list to hold our results
eGLM<-list()
par(mfrow=c(2,3))

for (i in 1:folds) {
  train_glm <- presence[kfold_pres!= i,]
  test_glm <- presence[kfold_pres == i,]
  backTrain_glm<-absence[kfold_back!=i,]
  backTest_glm<-absence[kfold_back==i,]
  dataTrain_glm<-rbind(train_glm,backTrain_glm)
  dataTest_glm<-rbind(test_glm,backTest_glm)
  glm_eval_2 <- glm(Presence~.,binomial(link = "logit"), data=dataTrain_glm)
  eGLM[[i]] <- evaluate(p=dataTest_glm[ which(dataTest_glm$Presence==1),],a=dataTest_glm[which(dataTest_glm$Presence==0),], glm_eval_2)
  
  #check the AUC by plotting ROC values
  
  plot(eGLM[[i]],'ROC')
  
}
summary(glm_eval_2)
eGLM

#testing
y2=dataTest_glm$Presence

prGLM2 <- predict(glm_eval_2,dataTest_glm)

evaluate(all_data2$Presence, prGLM2)

# calculate RMSE
rmse_glm2=RMSE(y2,prGLM2)
rmse_glm2 

#AUC
aucGLM <- sapply( eGLM, function(x){slot(x, 'auc')} )

mean(aucGLM)

Opt_GLM<-sapply( eGLM, function(x){ x@t[which.max(x@TPR + x@TNR)] } )

Opt_GLM

#take the mean to be applied to our predictions

Mean_OptGLM<- mean(Opt_GLM)

trGLM<-plogis(Mean_OptGLM)

trGLM

prGLM <- predict(data, glm_d,type = "response")
par(mfrow=c(1,2))
plot(prGLM, main='GLM, regression')
plot(pol,add=TRUE,col="transparent",border='dark grey')
plot(prGLM > trGLM, main='presence/absence')
plot(pol,add=TRUE,col="transparent",border='dark grey')
prGLM_pa=prGLM > trGLM
writeRaster(x=prGLM_pa,filename = "generalized_pa.tif",overwrite=TRUE)


##### 2- Maxnet  ##############################################################

eMAX<-list()
par(mfrow=c(2,3))
for (i in 1:folds) {
  train_maxnet <- presence[kfold_pres!= i,]
  test_maxnet <- presence[kfold_pres == i,]
  backTrain_maxnet<-absence[kfold_back!=i,]
  backTest_maxnet<-absence[kfold_back==i,]
  dataTrain_maxnet<-rbind(train_maxnet,backTrain_maxnet)
  dataTest_maxnet<-rbind(test_maxnet,backTest_maxnet)
  maxnet_eval_2 <- maxnet(dataTrain_maxnet$Presence,dataTrain_maxnet[,2:8])
  eMAX[[i]] <- evaluate(p=dataTest_maxnet[ which(dataTest_maxnet$Presence==1),],
                        a=dataTest_maxnet[which(dataTest_maxnet$Presence==0),], 
                        maxnet_eval_2)#use testing data (kfold==i) for model evaluation
  
  #check the AUC by plotting ROC values
  
  plot(eMAX[[i]],'ROC')
  
}
?maxnet

#inspect

eMAX

aucMAX <- sapply( eMAX, function(x){slot(x, 'auc')} )

#calculate the mean values for comparison with other models

aucMAX

mean(aucMAX)

#Get maxTPR+TNR for the maxnet model

Opt_MAX<-sapply( eMAX, function(x){ x@t[which.max(x@TPR + x@TNR)] } )

Opt_MAX

Mean_OptMAX<-mean(Opt_MAX)

Mean_OptMAX

prMAX <- predict(data, maxnet_eval_2)
par(mfrow=c(1,2))
plot(prMAX, main='Maxent Prediction')
plot(pol,add=TRUE,col="transparent",border='dark grey')
plot(prMAX > Mean_OptMAX, main='presence/absence')
plot(pol,add=TRUE,col="transparent",border='dark grey')
prMax_pa=prMAX > Mean_OptMAX
writeRaster(x=prMax_pa,filename = "maxnet_pa.tif",overwrite=TRUE)

###################### 3 - RF ##################################################

tuneRF(x=all_data2[,2:8],y=all_data2$Presence)

rf.all=randomForest(as.factor(Presence)~.,mtry=7,ntree=500,data=all_data2)

evaluate(all_data2$Presence,rf.all)#con todo el dataset
#set number of folds to use

folds=5

#partiction presence and absence data according to folds using the kfold() function.

kfold_pres <- kfold(presence, folds)
kfold_back <- kfold(absence, folds)

#create an empty list to hold our results (remember there will be five sets)
set.seed(825)
eRF<-list()
par(mfrow=c(2,3))

for (i in 1:folds) {
  train_rf <- presence[kfold_pres!= i,]
  test_rf <- presence[kfold_pres == i,]
  backTrain_rf<-absence[kfold_back!=i,]
  backTest_rf<-absence[kfold_back==i,]
  dataTrain_rf<-rbind(train_rf,backTrain_rf)
  dataTest_rf<-rbind(test_rf,backTest_rf)
  RF_eval <- randomForest(as.factor(Presence)~.,mtry=7,ntree=5000, data=dataTrain_rf)
  rf.pred <- predict(RF_eval, type="prob")[,2]#make prediction
  eRF[[i]]<-evaluate(p = rf.pred[which(dataTrain_rf$Presence == "1")], 
                     a = rf.pred[which(dataTrain_rf$Presence == "0")])
  
  #check the AUC by plotting ROC values
  
  plot(eRF[[i]],'ROC')
  
}
summary(RF_eval$importance)
RF_eval$importance

#inspect

eRF

RF_eval

aucRF <- sapply( eRF, function(x){slot(x, 'auc')} )

#calculate the mean values for comparison with other models
mean(aucRF) 

#Get maxTPR+TNR for the Random Forest model

Opt_RF<-sapply( eRF, function(x){ x@t[which.max(x@TPR + x@TNR)] } )

Opt_RF

Mean_OptRF<-mean(Opt_RF)

Mean_OptRF

prRF <- predict(data, RF_eval)


par(mfrow=c(1,2))
plot(prRF, main='Random Forest Prediction')
plot(pol,add=TRUE,col="transparent",border='dark grey')
tr <- threshold(eRF[[3]], 'spec_sens')
tr
plot(prRF > tr, main='presence/absence')
plot(pol,add=TRUE,col="transparent",border='dark grey')
prRF_pa=prRF > tr
writeRaster(x=prRF_pa,filename = "randomForest_pa2.tif",overwrite=TRUE)
writeRaster(x=prRF,filename = "randomForest2.tif",overwrite=TRUE)
