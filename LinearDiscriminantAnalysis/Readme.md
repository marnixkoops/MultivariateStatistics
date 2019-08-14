



**INTRODUCTION**

Households tend to have subscriptions to a particular magazine because
of their interests in certain areas. We have a data set of the US
Midwest containing information of households regarding magazine
subscriptions and other characteristics. With this data we try to answer
the research following question: We will investigate this using linear
discriminant analysis (LDA).

**DATA**

In this report, we analyze data of 141 households from a suburban panel
in the US Midwest with a subscription to one of 4 magazines. The
included variables in the dataset are: and , where *magazines* is used
as grouping variable and has 4 classes: Better Home & Gardens, Readers
Digest, TV-Guide and Newsweek. The dataset includes numerical ( and ),
ordinal (such as ) and nominal variables (such as ). LDA requires
numerical data, so we transform all variables to numerics. This works
because the nominal variables are binary and the ordinal variables have
a level structure which is preserved when transforming to numeric. The
groups are relatively balanced with 26 observations for Better Home &
Gardens, 49 for Readers Digest, 39 for TV-Guide and 27 for Newsweek. The
group samples are relatively small for 10 variables, meaning there is a
high chance that the within group covariance matrix could be
rank-deficient. Also the total sample of 141 observations is relatively
small in order to obtain reliable and consistent results.

**METHODOLOGY**

With linear discriminant analysis (LDA) one attempts to identify the
differences in characteristics between two or more groups. For instance
students and non-students, or households that read different magazines.
LDA is also known as canonical discriminant analysis or simply
discriminant analysis. Multiple approaches exists to perform LDA such as
Fisher’s, maximum likelihood (ML) or Mahalanobis. We use Fisher’s
approach to mathematically introduce LDA. The objective is to
discriminate between groups by finding the best possible separation. The
objective of this technique is to discriminate between a number of
groups, *G*, by finding *G* − 1 linear combinations
**t**<sub>*s*</sub> = **X****k**<sub>*s*</sub> of the characteristics
that best describe the different groups. If we have a *n* × *m* data
matrix **X** that contains *n* observations for *m* variables, the *G*
groups are separated by (*G* − 1) lines, planes, or hyperplanes
(depending on the dimensionality) that is set by the amount of variables
*m*. The *m* × (*G* − 1) matrix **K** contains all column vectors
**k**<sub>*s*</sub>. These separators are found by maximizing equation
() in order to obtain vectors **k**<sub>*s*</sub> for each separator
*s*.

Here, **A** is the across-group covariance matrix computed by
$\\mathbf{A} = (n-1)^{-1} {(\\mathbf{GM}-\\mathbf{1} {\\bar{\\mathbf{x}}}')}'(\\mathbf{GM}-\\mathbf{1} {\\bar{\\mathbf{x}}}')$,
and **W** is the within-group covariance matrix computed by
**W** = (*n* − 1)<sup> − 1</sup>(**X** − **G****M**)′(**X** − **G****M**).
Note that we assume homogeneity, that is we assume that the within group
covariance matrices are the same for each group. In these two covariance
equations, **M** is an *m* × *m* matrix that contains the means of all
variables per group. To indicate which observations belong to which
group mean, the *n* × *m* indicator matrix **G** is multiplied with
**M**. For each group *k* there is a dummy variable so that for each
observation the appropriate group mean is selected. The group means are
determined by taking the mean of all observations *i* in group *k*,
giving **M** = (**G**′**G**)<sup> − 1</sup>**G**′**X****K**. The 1 × *m*
vector ${\\bar{\\mathbf{x}}}'$ is obtained by calculating the mean of
each variable *j* for all observations *i*. The aim of maximizing () is
to minimize the distance between observations and the group mean within
a group (denominator of ()), and to maximize the distance between the
group means of all groups (numerator of ()). By setting the
**K**′**W****K** equal to the (*G* − 1) × (*G* − 1) identity matrix
**I**, () maps all ellipse shaped within-group covariance boundaries
around the observations, into circles.  
When choosing
$\\mathbf{b}\_{s} = \\mathbf{W}^{-\\frac{1}{2}} \\mathbf{k}\_{s} \\text{ with } {\\mathbf{b}\_{s}}'\\mathbf{b}\_{s}=1$,
equation () can be rewritten as
$L(\\mathbf{K}) = \\sum\_{s=1}^{G-1} {\\mathbf{b}\_{s}}' \\mathbf{W}^{-\\frac{1}{2}} \\mathbf{A} \\mathbf{W}^{-\\frac{1}{2}} \\mathbf{b}\_{s} = \\sum\_{s=1}^{G-1} {\\mathbf{b}\_{s}}' \\mathbf{E} \\mathbf{b}\_{s} = \\textup{tr } {\\mathbf{B}}'\\mathbf{EB}$.
This can now simply be solved by the eigendecomposition
**E** = **B****Γ****B**′. Here, **Γ** shows the eigenvalues that
indicate the importance of discrimination in reduced space. The
separators now obtain weights
$\\mathbf{k}\_{s}=\\mathbf{W}^{-\\frac{1}{2}} \\mathbf{b}\_{s}$ and
these vectors combined form matrix
$\\mathbf{K}= \\mathbf{W}^{-\\frac{1}{2}} \\mathbf{B}$. The eigenvalues
give us the proportion of trace which can be used as a diagnostic
measure. This proportion reports the amount of explained between-group
variance by each linear discriminant (LD). For prediction purposes all
-1 dimensions are used.

To predict the class of an observation the decision rule
$\\textit{predicted class} = \\textup{argmax}\_k \\boldsymbol{\\delta }\_{k} (\\mathbf{x})$
can be used with linear discriminant function
$\\boldsymbol{\\delta }\_{k} (\\mathbf{X}) = {\\mathbf{X}}' \\boldsymbol{\\Sigma }^{-1} \\boldsymbol{\\mu}\_{k} -\\frac{1}{2} {\\boldsymbol{\\mu }\_{k}}' \\boldsymbol{\\Sigma }^{-1} \\boldsymbol{\\mu}\_{k} + \\textup{log } \\boldsymbol{\\pi}\_{k}$.
Here, given an observation **x**<sub>*i*</sub>, and given
$\\hat{\\boldsymbol{\\mu}}\_{k} = \\sum\_{g\_{i}=k} \\mathbf{x}\_i$ and
$\\hat{\\pi}\_{k}=\\frac{n\_{k}}{n}$, one can compute
$\\hat{\\mathbf{\\Sigma}} = (n-(G-1))^{-1} \\sum\_{g\_{i}=k} \\sum\_{k=1}^{G-1} (\\mathbf{x}\_{i} - \\boldsymbol{\\mu}\_{k}) {(\\mathbf{x}\_{i} - \\boldsymbol{\\mu}\_{k})}'$.
Here, **Σ̂** is the pooled covariance matrix between the variables of the
observation **x**<sub>*i*</sub> and the variables of each group. The
decision rule assigns **x**<sub>*i*</sub> to the appropriate group *k*,
by choosing *k* for which the following function is minimized:
$\\textup{argmin}\_{k} {(\\mathbf{x}\_{i} - \\hat{\\boldsymbol{\\mu}}\_{k})}' \\hat{\\boldsymbol{\\Sigma}}^{-1} (\\mathbf{x}\_{i} - \\hat{\\boldsymbol{\\mu}}\_{k})= {(\\mathbf{x}\_{i} - \\hat{\\boldsymbol{\\mu}}\_{k})}' \\mathbf{U} \\mathbf{D}^{-1} {\\mathbf{U}}' (\\mathbf{x}\_{i} - \\hat{\\boldsymbol{\\mu}}\_{k})$,
where the eigen decomposition
$\\hat{\\boldsymbol{\\Sigma}} = \\mathbf{UD} {\\mathbf{U}}'$ is used.

To implement LDA we use the *M**A**S**S* package in R. We expect that
some variables have a greater influence on the discriminative power in
the model than others. Therefore, we will perform a variable selection
after considering the results. We will try to preserve the predictive
power while aiming for a parsimonious model. This is done by considering
the *R*<sup>2</sup> and p-values of the variables with a 5% critical
level. In addition, we assess the predictive power of the obtained
models. Predictive accuracy is determined by a leave one out
cross-validation test as the total amount of observations is relatively
small. Here -1 observations are used as a training set to predict the
left out observation. After the frequency of correct predictions gives
us the hit-rate (or accuracy), which is used as diagnostic for the
predictive power. If the groups would have been less balanced, the
accuracy would not be a suitable diagnostic and measures such as
precision and recall are more appropriate. But in this application the
hit-rate is sufficient. After selecting a model we interpret the results
by use of a biplot of the most important dimensions. Here we will plot
the object discriminant scores **X****K** as points, and the class
centroids **M** as large points. Then we will let the variables be
represented by arrows where the pooled within group correlations between
**X****K** and the component loadings **X**, calculated by
*n*<sup> − 1</sup>**X**′(**I** − **G**(**G**′**G**)<sup> − 1</sup>**G**)′**X****K**.
Then by comparing the direction and size of the arrows compared to the
group centroid locations we will be able to draw conclusions on whether
a variable discriminates well (achieves separation) for a certain group.

**RESULTS**

First, we apply LDA on the entire set of 11 variables. In sample
accuracy (hit-rate) is 50.35% as reported in Table 1 on the left. In
addition, we find that not all variables are equally important to
seperate the magazine groups. Therefore, we try to obtain a parsimonious
model while preserving accuracy by selecting only the most important
variables based on *R*<sup>2</sup> and p-value. In addition, focusing on
the most important features can lead to a more convenient
interpretation. A forward variable selection procedure results in the
selection of the following variables: *income*, *no male household
head*, *age* and *education*. We are aware that this procedure is greedy
and does not ensure an optimal selection but for now it suffices.
In-sample accuracy is 44.68% as reported in Table 1 on the right. Adding
*no female household head* results in a model with an in-sample accuracy
of 47.52% where both *no female household head* and *age* are
insignificant based on p-value. Hence, we decide the model with 5
variables is not worth the small increase in accuracy and continue with
the model as reported in the right of Table 1.
It is interesting to look at the actual linear discriminant (LD)
functions in the model. The following linear combinations of variables
constitute the functions:

-   `LD1 = -0.82 * Income +0.03 * NoMaleHouseHold +0.08 * Age -0.39 * Education`
-   `LD2 = -0.22 * Income -1.10 * NoMaleHouseHold +0.06 * Age -0.26 * Education`
-   `LD3 = 0.66 * Income -0.03 * NoMaleHouseHold +0.97 * Age -0.40 * Education`

We find *income* is most important in LD1, *no male houshold head* in
LD2 and *age* in LD3 as the values are scaled and centered. Furthermore,
the proportion of trace can be used to check separation achieved by the
discriminant functions. We find LD2 is more important when using all
variables than in the small model where LD1 accounts for 75%, LD2 for
18% and LD3 for 7% of the separation. This is in agreement with the
findings in the biplot in Figure 1 where we observe more horizontal
separation (LD1) between the groups than vertical (LD2). For instance by
looking at Newsweek and Readers Digest.

Next, we analyze the results by means of a biplot shown in Figure 1. In
the biplot of LD1 and LD2 we see that having *no male household head*
correlates negatively with having Better Home & Gardens in house.
*Education* correlates negatively with having a TV-Guide and in lesser
sense to having Readers Digest, but it seems to correlate strongly in
positive direction with a subscription to Newsweek. *income* is highly
correlated with *education* as can be seen from the biplot, and
therefore has comparable correlations. *Age* has relatively low
correlations, but we observe that older people tend to subscribe to
Readers Digest. Plotting LD1 and LD3 does not lead to very different
results, as the relation between the arrows and centroids remain mostly
the same, only *income* lies closer to Better Home & Gardens. When
plotting LD2 and LD3 we find that the centroids are very close together
making it hard to draw substantive conclusions.

Lastly, we examine predictive power. The goal is often to classify new
observations after training on labeled data known as supervised
clustering. Hence, we are interested in the predictive performance of
our model. We compare performance of a full model with all variables
included and the small model after variable selection. To increase
reliability we employ leave-one-out cross-validation (CV). The *l**d**a*
function in the MASS package has the ability to perform CV but for
educational purposes we have coded this ourselves. Results of the models
are compared in Table 2. We conclude that selecting the 4 most important
variables out of 11 yields an equal predictive accuracy of 38.30% while
providing a more parsimonious model with easier interpretation. As a
bonus, and to get a better idea of the accuracy, we perform the same
prediction routine with a Random Forest (RF) model. RF on the selected
variables obtains an accuracy of 29.79%, meaning both our LDA models
yield better predictive performance on this dataset. An RF with all
variables included performs worse with an accuracy of 25.53%. (We must
note that the RF is performed without extensive parameter tuning). In
all models we observe a large amount of misclassification for group 2
Readers Digest, while the true magazine is 3, TV-Guide. Apparently,
these magazines are hardest to separate based on the information in the
data. This finding is supported by the proximity of the centroids of
these groups in the biplot.

**CONCLUSION & DISCUSSION**

To answer our research question, , we have focused on the most
discriminating variables *income*, *no male house hold head*, *age* and
*education*. Based on the biplot of the correlations and the group
centroids we can conclude that younger households with high income and
education are more likely to be subscribed to Newsweek. These households
will also be less likely to have a subscription on TV-Guide and Readers
Digest. In contrast, we find that older households with lower income are
more likely to subscribe to Readers Digest. Further, we find households
with *no male household head* are less likely to be subscribed to Better
Home & Gardens.

We want to emphasize that care should be taken with the results and
conclusions in this report. The hit-rate we have found varies from 7%
for group 1 to 65% for group 2. In total, a hit-rate of below 40% is a
relative low accuracy when used for the purpose of forecasting. In
addition, as indicated in the data section a higher amount of
observations would provide more consistent results. Although the default
LDA function in *M**A**S**S* does yield a solution, the covariance
matrix of the first group is not invertible. This firstly leads us to
rejecting the assumption of equal covariance matrices and secondly poses
as a barrier to compute the solutions with Fisher’s or Mahanolobis’s
approach. When the interest is actionable insights, we suggest that the
model is extended with more data to draw more substantive and reliable
conclusions on the relation between demographic profiles of households
and their magazine subscription.

![Biplot of LD1 against LD2 of the small model. Large points indicate the group means where BH&G denotes Better Home & Gardens, RD is Readers Digest, TV-G is TV-Guide and NW is Newsweek.](Github_LinearDiscriminantAnalysis_files/figure-markdown_github/unnamed-chunk-5-1.pdf)

**APPENDIX**

**A: Comparison of LDA and QDA partitioning**

Out of interest we compare a linear and Quadratic Discriminant Analysis
(QDA) model. QDA drops the assumption of equal covariance matrices
across groups which is more flexible but also more sensitive to
differences in covariances than LDA. Often the results are similar and
we want to confirm this. Therefore, we visualise the partitioning of the
selected variables with LDA in Figure 2 and QDA in Figure 3, correct
classes are black numbers and misclassified ones are red. As expected,
the results are comparable in terms of partitioning and error. No
further interpration is done here.

![Partition plot of LDA on selected variables. Correct classes are black numbers and misclassified ones are red](Github_LinearDiscriminantAnalysis_files/figure-markdown_github/unnamed-chunk-6-1.pdf)

![Partition plot of QDA on selected variables. Correct classes are black numbers and misclassified ones are red](Github_LinearDiscriminantAnalysis_files/figure-markdown_github/unnamed-chunk-7-1.pdf)

**B: Code**

*Creating the biplot*

``` r
# Function to plot biplot
biplot_lda <- function(x,a,b){
  
  g<-length(x$res$lev)      # number of groups
  p<-dim(x$X)[2]            # number of variables
  ndim <- g-1               # number of dimensions
  xlab <- paste("LD",a,sep="")
  ylab <- paste("LD",b,sep="")
  title <- paste("Biplot on",xlab, "and",ylab,sep=" ")
  # plot object scores
  plot(x$Ds[, a], x$Ds[, b], type = "p", pch = c(0,1,2,5)[x$group], cex=1,
       col = x$col.group[x$group],main=title,xlab=xlab,ylab=ylab, 
       las = 1, cex.axis=0.8, cex.lab=0.8, xlim=c(-3,3), ylim=c(-3,3), asp=1)
  G <- 0 + outer(x$group, 1:4, "==")# construct g*n indicator matrix with dummies
  M <- diag(1/colSums(G)) %*% t(G) %*% x$Ds # Compute centroids for groups
  points(M[, a], M[, b], type = "p", pch = c(15,16,17,18), 
         cex = 2, col = x$col.group) #plot centroids
  text(M[, a], M[, b], labels = group_names_short, pos = 3, 
       col="dimgrey") # Add short names to plot
  
  # construct arrows and plot them together with labels  
  arrows(rep(0,p),rep(0,p),x$comp.load[,a], x$comp.load[,b],length = 0.05)
  posvec <- apply(x$comp.load, 1, sign)[2,] + 2
  text(comp.load, labels = row.names(x$comp.load), pos = posvec, 
       cex = 0.8, col = "black")
  legend("topright", group_names_short, pch = c(0,1,2,5), col=x$col.group)
}

# Actually plot the biplot using the above function
data = sapply(magazines, as.numeric) # Factors are leveled in correct order
X = scale(data[ ,c(3,7,10,11)], center = TRUE, scale = TRUE) # Scale data
res = lda(X, grouping = data[ ,1]) # Perform LDA with magazines as grouping var
Ds = X %*% res$scaling # Compute discrimination scores for objects
group  = as.numeric(data[, 1]) # numeric grouping vector
col.group = c("#F8766D", "#00BA38", "#619CFF", "#FD61D1")
group_names = levels(unique(magazines[,1]))
group_names_short = c("BH&G","RD","TV-G", "NW")
G = 0 + outer(group, 1:4, "==")
M = diag(1/colSums(G)) %*% t(G) %*% Ds # Compute centroids for groups
# Compute pooled within group correlations of predictors and discriminant scores
X.minGroupMean = X - G %*%  diag(1/colSums(G)) %*% (t(G) %*% X)
comp.load = cor(X.minGroupMean, Ds)
res_lda = list(X=X,res=res,Ds=Ds,group=group,col.group=col.group,
               comp.load=comp.load,group_names_short=group_names_short)
biplot_lda(res_lda,1,2) # Plot biplot of LD1 vs LD2
```

*Examine predictive accuracy with leave-one-out CV*

``` r
data_sel = sapply(magazines[ ,c(1,3,7,10,11)], as.numeric) # transform data
group = as.numeric(data_sel[, 1]) # save grouping var
n = dim(data_sel)[1]
# Factors are already labeled in correct order
data_scaled = scale(data_sel[ ,-1], center = TRUE, scale = TRUE) # scale data
data_scaled = cbind(group, data_scaled) # add back grouping var

trueclass = numeric(0)
testclass = numeric(0)
test_lda = list()

# Leave one out prediction
for(i in 1:n){
  train = data_scaled[-i, ] # create train set
  test = as.matrix(data_scaled[i, ]) # create test set
  trueclass[i]=test[1]
  # train lda on training set
  train_lda = lda(train[ ,-1], grouping = train[ ,1])
  
  # predict with lda on test set
  test_lda[[i]] = predict(object = train_lda, newdata=test[-1])
  testclass[i] = as.numeric(test_lda[[i]]$class)
}

hitrate_sel = length(which(testclass==trueclass))/n # calculate hit-rate
require(caret)
conf_sel = confusionMatrix(testclass, trueclass) # obtain full diagnostics
```

*Random Forest Predictions*

``` r
suppressMessages(library(randomForest)) # load package
set.seed(15102017) # set seed for reproducibility

data_sel = sapply(magazines[ ,c(1,3,7,10,11)], as.numeric) 
group = as.numeric(data_sel[, 1]) # save grouping var
n = dim(data_sel)[1]
# Factors are already labeled in correct order
data_scaled = scale(data_sel[ ,-1], center = TRUE, scale = TRUE) # Scale data
data_scaled = cbind(group, data_scaled) # add back grouping var

trueclass = numeric(0)
testclass = numeric(0)
test_rf = list()
# Leave one out prediction
for(i in 1:n) {
  train = data_scaled[-i, ] # create train set
  test = as.matrix(data_scaled[i, ]) # create test set
  trueclass[i]=test[1]
  # train lda on training set
  train_rf = randomForest(x=train[ ,-1], y=as.factor(train[ ,1]), 
                          type="class", importance=T, ntree=500)
  
  # predict with lda on test set
  test_rf[[i]] = predict(object = train_rf, newdata=test[-1])
  testclass[i] = as.numeric(test_rf[[i]])
}

hitrate_rf = length(which(testclass==trueclass))/n
conf_rf = confusionMatrix(testclass, trueclass)
```
