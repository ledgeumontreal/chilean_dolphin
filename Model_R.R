## Paper Mapping the Future: Revealing Habitat Preferences and Patterns of the 
##dangered Chilean Dolphin in Seno Skyring, Patagonia

#### PERFORMING DIFFERENT METHODS TO PREDICT PRESENCE/ABSENCE DAUPHIN CHILIEN-
####                            SKYRING

# Liliana Perez, Yenny Cuellar, Jorge Gibbons, Elias Pinilla Matamala, 
# Simon Demers, Juan Capella

## Paper Mapping the Future: Revealing Habitat Preferences and Pat-terns of the 
##dangered Chilean Dolphin in Seno Skyring, Patagonia

#### PERFORMING DIFFERENT METHODS TO PREDICT PRESENCE/ABSENCE DAUPHIN CHILIEN-
####                            SKYRING

# Liliana Perez, Yenny Cuellar, Jorge Gibbons, Elias Pinilla Matamala, 
# Simon Demers, Juan Capella

#setting libraries
library(raster)
library(sf)
library(randomForest) ## Random Forestm method
library(tibble)
library(dismo)
library(ModelMetrics)
library(nnet)
library(caret)

#reading data
bathymetry=raster("bathymetry.tif")
fish_farming=raster("fish_farming.tif")
kelp=raster("kelp.tif")
oxygen_dissolved=raster("oxygen_dissolved.tif")
salinity=raster("Salinity.tif")
silice=raster("silice.tif")
temp_seafloor=raster("temperature_seafloor.tif")
obs_csv=read.csv("observations.csv", sep = ";")
rios=raster("rios_distance.tif")
dist_coast=raster("distance_shoreline.tif")
turbidity=raster("turbidity.tif")
## shp study area
pol=read_sf("polygonSkyring.shp")
coordinates(obs_csv)= ~ x + y ## TO SET x and y as coordinates
#Assembling all predictor variable values into one RasterBrick
datall=stack(bathymetry,fish_farming,kelp,oxygen_dissolved,salinity,silice,temp_seafloor,rios, turbidity,dist_coast)
#print(names(datall))
names(datall)[1] <- "bathymetry"
names(datall)[2] <- "Fish farms"
names(datall)[5] <- "salinity"
names(datall)[8] <- "rios"
names(datall)[9] <- "turbidity"
names(datall)[10] <- "dist_coast"
#extract raster value by points
#all variables
rasvalueall=extract(datall,obs_csv)

combinePointValueall=cbind(obs_csv,rasvalueall)
all_data=as.data.frame(combinePointValueall)
drops <- c("FID","Shape..","POINTID","y","x") #taking fields to delete
all_data2=all_data[ , !(names(all_data) %in% drops)] #to obtain just the fields of interest




################################### Random Forest ###############################
set.seed(5)
tuneRF(x=all_data2[,2:11],y=all_data2$Presence)

presence=subset(all_data2,Presence==1)
absence=subset(all_data2,Presence==0)

rf_all <- randomForest(as.factor(Presence) ~.,mtry=6,ntree=5000, data=all_data2)
rf_all$importance

#MeanDecreaseGini
# bathymetry                  35.200533
# Fish.farms                  50.342678
# kelp                         1.391421
# oxygen_dissolved            26.225336
# salinity                    48.145211
# silice                      19.661473
# temperature_seafloor        21.503170
# rios                        28.005266
# turbidity                   31.037959
# dist_coast                 111.329990

sorted_importance <- sort(rf_all$importance, decreasing = TRUE)
sorted_importance
#set number of folds to use
folds=5

#partition presence and absence data according to folds using the kfold() function.

kfold_pres <- kfold(presence, folds)
kfold_back <- kfold(absence, folds)

#create an empty list to hold our results (remember there will be five sets)

eRF<-list()
eRFtr<-list()
par(mfrow=c(3,3))

for (i in 1:folds) {
  train_rf <- presence[kfold_pres!= i,]
  test_rf <- presence[kfold_pres == i,]
  backTrain_rf<-absence[kfold_back!=i,]
  backTest<-absence[kfold_back==i,]
  dataTrain_rf<-rbind(train_rf,backTrain_rf)
  dataTest_rf<-rbind(test_rf,backTest)
  RF_eval <- randomForest(formula=Presence~.,mtry=6,ntree=5000, data=dataTrain_rf,importance=TRUE)#this is our RF model
  rf.predall <- predict(RF_eval,dataTest_rf)#make prediction
  
  eRF[[i]]<-evaluate(dataTest_rf[dataTest_rf$Presence=="1",],
                     dataTest_rf[dataTest_rf$Presence=="0",],RF_eval )#validation test-AUC:
  eRFtr[[i]]<-evaluate(dataTrain_rf[dataTrain_rf$Presence=="1",],
                       dataTrain_rf[dataTrain_rf$Presence=="0",],RF_eval )
  }

#AUC
aucrftest <- sapply( eRF, function(x){slot(x, 'auc')} )
mean(aucrftest)
aucrftr <- sapply( eRFtr, function(x){slot(x, 'auc')} )
mean(aucrftr)

#Get maxTPR+TNR

Opt_RF<-sapply( eRFtr, function(x){ x@t[which.max(x@TPR + x@TNR)] } )

Mean_OptRF<-mean(Opt_RF)

# calculate RMSE

y=dataTest_rf$Presence
rmse_rf=rmse(y,rf.predall)

varImpPlot(RF_eval)

#Predict
pr_all <- predict(datall, RF_eval,type="response")

##PLOT
par(mfrow=c(1,2))
plot(pr_all, main='Random Forest')
plot(pol, col="transparent",add=TRUE, border='dark grey')
tr_all <- threshold(eRF[[3]], 'spec_sens') 

plot(pr_all > 0.51868, main='presence/absence') 
plot(pol, col="transparent",add=TRUE, border='dark grey')

#############  Generalized Linear Model ######################################

#create an empty list to hold our results (remember there will be five sets)
eGLM<-list()
eGLMtr<-list()
par(mfrow=c(2,3))

for (i in 1:folds) {
  train_glm <- presence[kfold_pres!= i,]
  test_glm <- presence[kfold_pres == i,]
  backTrain_glm<-absence[kfold_back!=i,]
  backTest_glm<-absence[kfold_back==i,]
  dataTrain_glm<-rbind(train_glm,backTrain_glm)
  dataTest_glm<-rbind(test_glm,backTest_glm)
  glm_eval_2 <- glm(Presence~.,binomial(link = "logit"), data=dataTrain_glm)#this is our glm model trained on presence and absence points
  eGLM[[i]] <- evaluate(p=dataTest_glm[ which(dataTest_glm$Presence==1),],a=dataTest_glm[which(dataTest_glm$Presence==0),], glm_eval_2)#use testing data (kfold==i) for model evaluation
  eGLMtr[[i]] <- evaluate(p=dataTrain_glm[ which(dataTrain_glm$Presence==1),],a=dataTrain_glm[which(dataTrain_glm$Presence==0),], glm_eval_2)#use testing data (kfold==i) for model evaluation
  
  #check the AUC by plotting ROC values
  
  plot(eGLM[[i]],'ROC')
  
}
summary(glm_eval_2)
eGLM
#?glm
#testing
y2=dataTest_glm$Presence

prGLM2 <- predict(glm_eval_2,dataTest_glm)

evaluate(all_data2$Presence, prGLM2)#con todo el dataset

# calculate RMSE
rmse_glm2=rmse(y2,prGLM2)
rmse_glm2 # 1.707303

#AUC test
aucGLM <- sapply( eGLM, function(x){slot(x, 'auc')} )

mean(aucGLM)#0.8115357
#AUC training
aucGLMtr <- sapply( eGLMtr, function(x){slot(x, 'auc')} )

mean(aucGLMtr)#0.8205

Opt_GLM<-sapply( eGLM, function(x){ x@t[which.max(x@TPR + x@TNR)] } )

Opt_GLM

#take the mean to be applied to our predictions

Mean_OptGLM<- mean(Opt_GLM)

trGLM<-plogis(Mean_OptGLM)

trGLM#0.4721455

prGLM <- predict(datall, glm_eval_2,type = "response")

par(mfrow=c(1,2))
plot(prGLM, main='GLM')
plot(pol,add=TRUE,col="transparent",border='dark grey')
plot(prGLM > trGLM, main='presence/absence')
plot(pol,add=TRUE,col="transparent",border='dark grey')


################### Neural Network ###########################################
########### Create training (70%) and test (30%) dataset ##############
set.seed(3)
data_split=initial_split(all_data2,prop = 0.7)
train=training(data_split) ## 70%
test=testing(data_split)  ## 30%
set.seed(5)
# set parameters for cross validation 
fitControl <- trainControl(## 5-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated 5 times
  repeats = 100)

## fit the model

neuronal=train(Presence~ ., data=train,
               trControl=fitControl,
               method="nnet",
               preProc=c("center","scale"),
               verbose=FALSE)

# testing  model results
##### Neuronal - make predictions to the test dataset 
pred_neuronal=predict(neuronal,newdata=test)
y=test$Presence
# calculate RMSE
rmse_neuronal=RMSE(y,pred_neuronal)
rmse_neuronal #0.373622

# calculate AUC

eval_neuronal_tr <- evaluate(train[train$Presence==1,],train[train$Presence==0,], neuronal)
eval_neuronal_tr #  AUC: 0.9424837 

eval_neuronal <- evaluate(test[test$Presence==1,],test[test$Presence==0,], neuronal)
eval_neuronal #AUC: 0.8842915   

## Making spatial predictions
neuronal_map=predict(datall,neuronal)

par(mfrow=c(1,2))
plot(neuronal_map, main='Neuronal prediction')
plot(pol,add=TRUE,col="transparent",border='dark grey')

plot(neuronal_map > 0.5970204   , main='presence/absence')
plot(pol,add=TRUE,col="transparent",border='dark grey')

