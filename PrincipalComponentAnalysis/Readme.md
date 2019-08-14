







<img src="Github_PrincipalComponentsAnalysis_files/figure-markdown_github/Plot sample images-1.pdf" alt="35 Random rows of the fashion MNIST data mapped into matrices and plotted." width="\textwidth" />
<p class="caption">
35 Random rows of the fashion MNIST data mapped into matrices and
plotted.
</p>

**INTRODUCTION**

Currently, data is accumulated at unprecedented rates resulting in large
and high dimensional datasets. When numerous variables are involved the
patterns can be complex. Consequently, the structure in a
multi-dimensional dataset is a rather abstract notion and difficult to
grasp. To improve comprehension of high dimensional datasets
dimensionality reduction can be applied. We are interested in modern
applications of dimensionality reduction in exciting fields such as
computer vision. Therefore, we analyse product images of e-commerce
company Zalando. An important realisation is the fact that an image is
represented by a collection of information like any other dataset. *The
goal of this research is to explore and analyse the patterns in these
product images.* In addition, we use the data structure to examine image
compression. The results can give rise to some interesting possiblities
such or automatic labeling of products in a webshop.

**DATA**

In this paper we analyse a dataset released by the research department
of Zalando \[Zalando, 2017\]. It contains 10,000 product images each of
the size 28 × 28 pixels. The dataset has been proposed to replace the
widely used MNIST dataset of handwritten digits to benchmark machine
learning algorithms. The images are converted to grayscale and each
observation is an array of 28 × 28 = 784 elements. Each element
describes the associated pixel-value indicating the darkness of that
pixel on a scale of 1 to 255 where 1 is a white pixel and 255 is black.
A row of pixel values can be mapped into a matrix and plotted to obtain
an image of that product. A random sample of 35 products is mapped and
plotted in Figure 1. In addition, each observation is labeled with a
group indicating the product type. The 10 included products groups are
*T-Shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag*
and *Boot*. To summarise, we have a 10, 000 × 785 data matrix where each
row describes a single product image with pixel values and associated
product group label.

**METHODOLOGY**

A proven technique is principal components analysis (PCA). The concept
originates from Pearson \[1901\] but was later formalized by Hotelling
\[1933\] and is still a cornerstone in modern statistical analysis. It
is currently applied in various fields such as computer vision
(e.g. facial recognition) and image compression. PCA is often used for
the following goals:

-   Explore and identify patterns in high dimensional data
-   Dimensionality reduction where *p*-dimensional data is mapped into a
    *k*-dimensional subspace with *k* ≤ *p*

This is achieved by finding a linear transformation of the variables to
capture a maximum amount of spread in the points. Hence, a principle
component (PC) is some linear combination of the original variables and
represents a direction through the data. The first PC captures the
largest possible variance. The second PC is independent of the first due
to orthogonality and captures the largest possible retaining variance
and so on. As a result, data can be compressed while preserving as much
of the original information as possible. The parts of the data with
highest variability in the data can be retained by selecting the first
*k* out of *p* components.

A convenient and efficient approach is employing Singular Value
Decomposition (SVD) to obtain the PCA solution as this yields
computation of all possible *p* components at once. A *n* × *p* data
matrix **X** can be decomposed as

where **U** is a matrix of eigenvectors of
$\\mathbf{X}\\mathbf{X}^\\textup{T}$ with
$\\mathbf{U}^\\textup{T}\\mathbf{U}=\\mathbf{I}$, **Σ** is the diagonal
matrix of singular values and $\\mathbf{V}^\\textup{T}$ is the matrix
with eigenvectors of $\\mathbf{X}^\\textup{T}\\mathbf{X}$ with
$\\mathbf{V}^\\textup{T}\\mathbf{V}=\\mathbf{I}$. Now, the columns of
**U** give the PC’s which are the new directions through the data. The
diagonal elements of **Σ** are proportional to the standard deviations
of the PCA. The component loadings are the coordinates of the datapoints
on basis of the new directions given by **U****Σ**.

The decomposition in (1) is influenced by the scale of measurement as
PCA aims to find a direction that maximizes variance. Therefore,
variables with high variance due to different scaling may undesirably
dominate the solution. This issue is negated by standardizing the input
data **X** such that the mean of each column is 0 and the variance is 1.
An attractive application is dimensionwise reconstruction with a
selection of *k* ≤ *p* dimensions of the original data. This yields the
dimension reduction ability of PCA. **X** can be reconstructed as

where **u**<sub>*s**p*</sub> is column *p* of **U** and
$\\mathbf{v}\_p^\\textup{T}$ is column *p* of **V**. For *k* = *p* we
have $\\hat{\\mathbf{X}}=\\mathbf{X}$, the original data. The
reconstruction yields an optimal least-squares solution in *k*
dimensions. This principle can be used to create a PCA biplot for visual
inspection of the first two or three dimensions with highest variance.
The biplot loss function is
*L*(**C**, **E**) = \|\|**X** − **C****E**′\|\| where
$\\mathbf{C}=\\sqrt{n} \\textbf{U}\\mathbf{\\Sigma}$ and
**E** = *n*<sup> − 0.5</sup>**U****Σ**. Now **E** are z-scores and **E**
are component loadings. The loss can be minimized by using the SVD of
**X** given in (1). Then, the inner product
$\\mathbf{c}\_i^\\textup{T} \\mathbf{e}\_j$ is visualised \[Groenen,
2010\].

Moreover, as we are analysing images, we use (2) to reconstruct and
visually examine the result of a lower dimensional representation of the
data. Next, The standard deviations of the PC’s given by
*d**i**a**g*(**Σ**) can be used to to calculate the proportion of
variances explained by each PC. Plotting yields a screeplot which can be
used as diagnostic tool to choose the number of dimensions to retain. A
rule of thumb is choosing *k* before the elbow point, sometimes a clear
elbow can not be determined. Another criterion is Kaiser’s rule which
suggests keeping only those components with an eigenvalue larger than
one.

Lastly, we apply the bootstrap to quantify the quality of our solution
\[Efron, 1986\]. Bootstrapping is a method to assess statistical
accuracy and can be broken down into the following general steps

-   Randomly draw a set of observations from the data with replacement,
    repeat *B* times
-   Analyse the behavior of the statistic of interest over the *B*
    different datasets, for instance compute a 95% confidence interval

An advantage of bootstrapping is that it is non-parametic, hence
requires no assumptions on the distrubution. However, it is often
computationally intensive. We will apply the bootstrap to obtain the
standard deviation of both the eigenvalues and component loadings in
order to examine the stability of our PCA solution. We use the SVD
approach in our PCA, hence we also use this for the boostrap.

**RESULTS**

First, we examine a screeplot to determine the amount of principal
components to retain. It reveals a clear elbow at *p* = 3 as seen in
Figure 2. Hence, we select the first two PC’s and obtain a solution
containing two out of the original 784 dimensions. This is an
exceptional reduction and a very interesting result as merely two
directions are needed to describe more than one third (36%) of the total
variability in the data.

<img src="Github_PrincipalComponentsAnalysis_files/figure-markdown_github/screeplot-1.pdf" alt="Screeplot of variance per component, elbow observed at PC 3."  />
<p class="caption">
Screeplot of variance per component, elbow observed at PC 3.
</p>

<img src="Github_PrincipalComponentsAnalysis_files/figure-markdown_github/bootstrap results-1.pdf" alt="Eigenvalues with bootstrapped standard deviations (red bars indicate range)"  />
<p class="caption">
Eigenvalues with bootstrapped standard deviations (red bars indicate
range)
</p>

<img src="Github_PrincipalComponentsAnalysis_files/figure-markdown_github/Visualize PCA-1.pdf" alt="Plot of first two dimensions of PCA solution."  />
<p class="caption">
Plot of first two dimensions of PCA solution.
</p>

Next, we visualize the first two dimensions of the PCA solution. A
traditional PCA biplot with all observations and variables proves less
helpful on this data. Adding the 784 variables as vectors overloads the
plot preventing reading of the points or labels as can be seen in Figure
6 in Appendix A. A line of interpretation here is that some products
score highly on pixel values. Hence, the biplot reveals specific or
groups of high or low pixel values that are characteristic for distinct
product groups. For instance, a *trouser* is more likely to have dark
(high) pixel-values in te top part of the image whereas a *sandal* is
more likely to have low pixel-values. Figure 4 plots the first two
dimenions, for a better interpretation of the structure we remove the
variables and add confidence intervals per group with a 95% level based
on a multivariate normal distrution. We mostly find products with a
square-like shape such as *bags*, *boots*, *sandals* and *sneakers* on
the negative side of the first and second PC (the lower left quadrant).
On contrary, more rectangle-shaped products such as *t-shirts*, *coats*
and *pullovers* are on the positive side of the first PC. The products
with the most rectangle-like shapes such as *trousers* and *dresses* are
grouped in the top. *Shirts* seem to be hardest to distinguish based on
the first two dimensions as this group is the most scattered throughout
the plot. These findings are intuitive. For instance, *sneakers* and
*sandals* are harder to make apart then *bags* and *trousers*.
Naturally, these difference are also also rooted in the underlying pixel
data which is the variability exploited by PCA. The most outlying
observations from the general cloud are seen in the bottom right corner
and are further investigated in Appendix B.

Third, we apply the bootstrap to examine to examine statistical
accuracy. For the eigenvalues we apply SVD and disable computation of
the left and right singular vectors. We only calculate the singular
values to greatly increase efficiency of the computation as our data is
relatively large. Figure 3 reports the eigenvalues and bootstrapped
standard deviations by means of an error bar. We observe little
deviation over the 1000 bootstrap iterations.

<img src="Github_PrincipalComponentsAnalysis_files/figure-markdown_github/unnamed-chunk-2-1.pdf" alt="Product images reconstructed with varying amounts of dimensions."  />
<p class="caption">
Product images reconstructed with varying amounts of dimensions.
</p>

Lastly, we reconstruct and plot the products using (2), results are
shown in Figure 5. Analysing the reconstructed images based on a varying
amount of PC’s supports the finding that even with only 2 of the
original 784 dimensions in the data a large portion of variance can be
explained. Some products are somewhat recognisable with only 2
dimensions. If visual image quality is of importance, 2 dimensions
appear insufficient for display on a website.

Kaiser’s rule would conclude in retaining the first 79 components, this
is roughly 10% of the original dimensions. The corresponding
reconstruction is seen in the fourth column in Figure 5. Indeed, this
amount results in a decent image quality and product groups can be
visually seperated with high certainty. The last column displays
*k* = *p*, which yields the original data as described in the
Methodology section.

**CONCLUSION & DISCUSSION**

This research has reported the principal components analysis of a set of
product images of Zalando. We have looked for structure and examined the
effect of a lower dimensional representation on image quality. We
conclude that roughly one third of the variation in the data can be
described by the first two PC’s. This is enough to identify different
shaped products such as *bags*, *sneakers*, *boots* from more elongated
products such as *trousers* and *dresses*. However, two dimenions are
not sufficient to accurately recognize similarly shaped products. The
reconstructions also shows that a *dress* and *trouser* may be very hard
to distinguish just like a *boot* and *sandal* in a lower dimensional
representation. Nonetheless, the relatively large distance between for
example *bags* and *trouser* in the second PC in Figure 4 is also
reflected in the reconstruction. Even with the first two PC’s the vague
dark outlines are indicative enough to determine if the image concerns a
more square-shaped product like a *bag* or *sneaker* or more elongated
such as a *trouser* or *dress*. However, for a higher certainty of
distinction between all product groups more PC’s are required.
Bootstraping shows that the results of PCA on this data are stable.

Furthermore, the image quality in a lower dimensional representation of
the data based on the first two PC’s is not up to standards. The lack of
details prohibits actual use of such a compressed image on the website.
The solution using only half of the original data (*k* = 392) appears to
be very close to the original image while requiring only half the disk
space.

Another interesting application of these findings is the possiblity to
use a selection of the PC’s as input to greatly reduce the computational
burden of machine learning algorithms. For instance, for an algorithmic
image classifier. In this case the PCA functions as a preliminary data
transformation to reduce dimensionality and computational intensity. All
in all, the ability of PCA to find structure in this high-dimensional
image data based on a combination of merely two components is
remarkable.

**COMPARISON WITH STANDARD OUTPUT**

This section compares the obtained result from our own written functions
with the output of standard `R` output. We employ the `prcomp()`
function which uses the SVD approach like our own function. Therefore,
results should be equal. We compare each corresponding element of the
output. First we compare the scores. Second, the eigenvector matrix
(rotations) are compared. Third, the eigenvalues (squared standard
deviations). Lastly, we compare the proportion of explained variances
per component.

``` r
pca_own = pca_fun(as.matrix(fmnist[,3:786])) # perform PCA with own function
pca_standard = prcomp(fmnist[,3:786]) # perform PCA with default function
all(round(pca_own$scores, 6) == round(pca_standard$x, 6), TRUE) # scores
```

    ## [1] TRUE

``` r
all(round(pca_own$loadings, 8) == round(pca_standard$rotation, 8), TRUE) # evectors
```

    ## [1] TRUE

``` r
all(round(pca_own$eigenvalues, 10) == round(pca_standard$sdev^2, 10), TRUE) # evalues
```

    ## [1] TRUE

``` r
standard_vars = summary(pca_standard)$importance # obtain PC variances
all(round(pca_own$prop_var, 5) == round(standard_vars[2, ], 5), TRUE) # prop vars
```

    ## [1] TRUE

All test resolve to `TRUE`, meaning we can conclude that the PCA result
of our own function and the standard `R` output is exactly equal up to
at least 5 decimal points.

**REFERENCES**

Zalando SE Research **\[2017\]**. *Fashion MNIST dataset.* Retrieved
from Github
<a href="https://github.com/zalandoresearch/fashion-mnist" class="uri">https://github.com/zalandoresearch/fashion-mnist</a>.

Gower, J.C., Groenen, P.J.F., & van de Velden, M. **\[2010\]**. *Area
Biplots*, Journal of Computational and Graphical Statistics, 19:1, 46-61

Pearson, K. **\[1901\]**. LIII. *On lines and planes of closest fit to
systems of points in space*. The London, Edinburgh, and Dublin
Philosophical Magazine and Journal of Science, 2(11), 559-572.

Hotelling, H. **\[1933\]**. *Analysis of a complex of statistical
variables into principal components.* Journal of educational psychology,
24(6), 417.

Efron, B., & Tibshirani, R. **\[1986\]**. *Bootstrap methods for
standard errors, confidence intervals, and other measures of statistical
accuracy.* Statistical science, 54-75.

Groenen, P. J., & van de Velden, M. **\[2016\]**. *Multidimensional
scaling by majorization: A review.* Journal of Statistical Software,
73(8), 1-26.

Maaten, L. V. D., & Hinton, G. **\[2008\]**. *Visualizing data using
t-SNE.* Journal of Machine Learning Research, 9(Nov), 2579-2605.

**APPENDIX**

<!-- **A: Overloaded PCA biplot** -->
<!-- Figure 6 shows the overloaded PCA biplot with observations and variables. Labels are not plotted to somewhat reduce the large amount of overlap in the variables and because all variable names are simply a specific pixel location such as "pixel23" without much deeper meaning. -->
<!-- ```{r, fig.cap="Overloaded PCA biplot"} -->
<!-- fviz_pca_biplot(pca_standard, geom="point", habillage=fmnist$cat, col.var = "dimgrey", alpha.var=0.5, label="none") + scale_shape_manual(values=seq(48,57)) + theme(aspect.ratio=1) + labs(title="PCA Biplot", x="Principal Component 1 (22%)", y="Principal Component 2 (14%)") + geom_vline(xintercept=c(-0,0), linetype="dashed", size=0.3) + geom_hline(yintercept=c(-0,0), linetype="dashed", size=0.3) -->
<!-- ``` -->
**A: Outlying Products**

Figure 7 shows the furthest outlying products based on the PCA solution
seen in the bottom right corner of Figure 3. Four are *bags* and one is
a *pullover*. We find that these products fill the largest portion of
the matrix whereas other products generally use a smaller portion of the
space. Hence, this difference is reflected in the underlying data. These
outlying products have higher pixel-values located in the border
positions whereas for instance a *trouser* tends to have light pixels
along both sides and a *sandal* tends to have light pixels in the top
portion of the matrix. This is reflected in the PCA solution.

<img src="Github_PrincipalComponentsAnalysis_files/figure-markdown_github/unnamed-chunk-4-1.pdf" alt="Outlying products observed when plotting the first two PC's." width=".5\linewidth" />
<p class="caption">
Outlying products observed when plotting the first two PC’s.
</p>

**B: Comparison of PCA and t-SNE**

A more recent dimensionality reduction technqiue is the t-Distributed
Stochastic Neighbor Embedding (t-SNE) algorithm \[Maaten, 2008\]. The
main goal is to visualize a high-dimensional structure with a faithful
representation in lower-dimensional space. The core principle can be
summarized as assigning each data-point to a location in a two or
three-dimensional map. Key-driver in the transformation from high to
lower dimensionality is the Kullback-Leibler divergence which is based
on the entropy measure from the field of information theory. T-SNE has
been shown to yield a better visualisation performance than PCA in
various cases \[Maaten, 2008\]. Out of interest we compare the solution
obtained by t-SNE with the solution of PCA by visualising the first two
dimensions shown in Figure 8. The ellipses indicate 95% confidence
intervals per group based on a multivariate normal distribution.

Both techniques show a dense cluster for *trousers* and a large amount
of overlap in *bags*, *boots* and *sandals*. In addition *t-shirts* and
*shirts* are hard to separate in the first two PC’s. The most noticable
difference is that t-SNE manages to obtain a clustering structure with
lower intra-cluster and higher inter-cluster distances. In other words,
we observe denser groups and a larger amount separation between the
groups. This indicates a better distinction can be made between the
product groups compared to PCA in the first two dimensions. Especially
*trousers* are almost completely isolated from other groups. In
addition, *t-shirts* form a very dense cluster but has some overlap with
*shirts*. This is logical, the only real difference is longer sleeves.
Interestingly, t-SNE manages to seperate *bags* from *boots* while these
groups show a very large amount of overlap in the PCA solution.

<img src="Github_PrincipalComponentsAnalysis_files/figure-markdown_github/unnamed-chunk-5-1.pdf" alt="Comparison of first two dimensions of PCA solution (left) and t-SNE (right). Ellipses indicate 95% confidence intervals per group." width=".49\linewidth" /><img src="Github_PrincipalComponentsAnalysis_files/figure-markdown_github/unnamed-chunk-5-2.pdf" alt="Comparison of first two dimensions of PCA solution (left) and t-SNE (right). Ellipses indicate 95% confidence intervals per group." width=".49\linewidth" />
<p class="caption">
Comparison of first two dimensions of PCA solution (left) and t-SNE
(right). Ellipses indicate 95% confidence intervals per group.
</p>

**CODE**

PCA function

``` r
pca_fun = function(x) { # function to perform PCA through SVD
  if (is.numeric(x) == F) { # check if input is strictly numeric
    print("ERROR: The input data must be numeric")
    return() # stop if data is not numeric
  }
  scale(x, center=T, scale=T) # center input data
  svd_res = svd(x) # decompose data matrix with SVD
  princomps = svd_res$u # obtain principal components
  loadings = svd_res$v # obtain loadings from SVD
  sigma = diag(svd_res$d) # calculate sigma matrix
  evalues = (svd_res$d / sqrt(nrow(x) - 1))^2 # calculate eigenvalues
  scores = x %*% loadings # calculate scores
  var = crosprod(sigma) # calculate variance
  tot_var = sum(diag(var)) # calculate total variance
  prop_var = diag(var) / tot_var # calculate prop variance
  # save results into list
  results_list = list("princomps"=princomps, "loadings"=loadings, "sigma"=sigma, 
                      "eigenvalues"=evalues,"scores"=scores, "var"=var, 
                      "tot_var"=tot_var, "prop_var"=prop_var)
  return(results_list) # return results
}
```

Screeplot (first calculate variances account for then plot)

``` r
# get proportional variances of our own PCA function
prop_var = as.data.frame(pca_own$prop_var[1:10])
colnames(prop_var) = "prop_var"
# make screeplot of first ten PC's
ggplot(prop_var, aes(x=c(1:10), y=prop_var)) + geom_bar(stat="identity", 
                                                        fill=c("#00B0F6", "#00B0F6", rep("dimgrey", 8))) + 
  labs(title="Screeplot of Principal Component 1 to 10", x="Principal Component", 
       y="% of Variance") + geom_line() + geom_point() + geom_text(size=3, 
                                                                   position=position_stack(vjust=0.75), label=c("22%", "14%", "0.05%", "0.05%", 
                                                                                                                "0.04%", "0.03%", "0.03%", "0.02%", "0.02%", "0.01%")) + 
  scale_x_continuous(breaks=c(1:10))
```

PCA plot (first two dimensions)

``` r
ggplot(data=c_zscores, aes(x=c_zscores[ ,1], y=c_zscores[ ,2], 
                           colour=fmnist$cat, shape=fmnist$cat))
+ geom_point(size=2) + theme(aspect.ratio=1) + scale_shape_manual(values=seq(48,57))
+ labs(title="PCA Solution", x="Principal Component 1 (22%)", 
       y="Principal Component 2 (14%)") + stat_ellipse(type="norm", level=0.95) 
+ geom_vline(xintercept=c(-0,0), linetype="dashed", size=0.3) 
+ geom_hline(yintercept=c(-0,0), linetype="dashed", size=0.3)
```

Bootstrap function and plot with standard deviations

``` r
input = fmnist[,c(3:786)] # input matrix for bootstrap

# bootstrapping eigenvalues
boot_evalues = function(data, index){ # function for extracting sigma's         
  svd_res = svd(input[index,], nu=0, nv=0) # only calculate d, no u and v (efficiency)
  return(svd_res$d) # return results                               
}

time = Sys.time()
boot_res = boot(input, boot_evalues, R = 1000) # run 1000 bootstraps
Sys.time() - time # report calculation time

orig_solution = as.data.frame(boot_res$t0[1:10]) # extract first 10 original PCs
colnames(orig_solution) = "eigenvalue" # rename column
bootstrap_sd = apply(boot_res$t[,c(1:10)], 2, sd) # bootstrapped SD's for 10 first PCs
# plot result for first 10 PCs
ggplot(orig_solution, aes(x=c(1:10), y=eigenvalue)) + 
  geom_bar(stat="identity", fill=c(rep("dimgrey", 8), "#00B0F6", "#00B0F6")) +
  geom_errorbar(aes(ymin=eigenvalue-bootstrap_sd, ymax=eigenvalue+bootstrap_sd), 
                width=0.5, color="#F8766D", size=0.5) + 
  labs(title="Eigenvalues of PC 1 to 10", 
       subtitle="Standard deviations marked with red errorbar", 
       x="Principal Component", y="Eigenvalue") + coord_flip() 
+ scale_x_reverse(breaks=c(1:10))
```

Reconstruction and plotting of the product images with *k* ≤ *p*
dimensions through SVD. Images are constructed by mapping a row of pixel
values into a 28 × 28 matrix and plotting the matrix.

``` r
svd_res = svd(fmnist[,3:786]) # perform SVD on data
comps = c(2, 5, 10, 25, 79, 157, 261, 392, 784) # choose set of dimensions
recon_list = list() # create placeholder list to save reconstructions
# reconstruct the data with different number of components
for (k in comps) { 
  x_hat = svd_res$u[,1:k] %*% diag(svd_res$d)[1:k,1:k] %*%
    t(svd_res$v[,1:k]) # dimensionwise reconstruction based on k
  recon_list[[length(recon_list)+1]] = x_hat # save reconstructions into list
}
# prepare data for mapping into matrix and setup plots
xy_axis = data.frame(x = expand.grid(1:28,28:1)[,1],
                     y = expand.grid(1:28,28:1)[,2])

plot_theme = list(raster = geom_raster(hjust = 0, vjust = 0), 
                  gradient_fill = scale_fill_gradient(low = "white", 
                                                      high = "black", guide = FALSE), 
                  theme(axis.line = element_blank(),axis.text = element_blank(),
                        axis.ticks = element_blank(),axis.title = element_blank(),
                        panel.background = element_blank(),panel.border = element_blank(),
                        panel.grid.major = element_blank(),
                        panel.grid.minor = element_blank(),
                        plot.background = element_blank())
)
# sample some selected products and map into a matrix to plot
sample_plots_dim2 = c(600,1199,604,206,210) %>% map(~ {plot_data = 
  cbind(xy_axis, fill=as.data.frame(t(as.data.frame(recon_list[[1]])[.x,]))[,1])
ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})
sample_plots_dim10 = c(600,1199,604,206,210) %>% map(~ {plot_data = 
  cbind(xy_axis, fill=as.data.frame(t(as.data.frame(recon_list[[3]])[.x,]))[,1])
ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})
sample_plots_dim25 = c(600,1199,604,206,210) %>% map(~ {plot_data = 
  cbind(xy_axis, fill=as.data.frame(t(as.data.frame(recon_list[[4]])[.x,]))[,1])
ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})
sample_plots_dim79 = c(600,1199,604,206,210) %>% map(~ {plot_data = 
  cbind(xy_axis, fill=as.data.frame(t(as.data.frame(recon_list[[5]])[.x,]))[,1])
ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})
sample_plots_dim261 = c(600,1199,604,206,210) %>% map(~ {plot_data = 
  cbind(xy_axis, fill=as.data.frame(t(as.data.frame(recon_list[[7]])[.x,]))[,1])
ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})
sample_plots_dim392 = c(600,1199,604,206,210) %>% map(~ {plot_data = 
  cbind(xy_axis, fill=as.data.frame(t(as.data.frame(recon_list[[8]])[.x,]))[,1])
ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})
sample_plots_dim784 = c(600,1199,604,206,210) %>% map(~ {plot_data = 
  cbind(xy_axis, fill=as.data.frame(t(as.data.frame(recon_list[[9]])[.x,]))[,1])
ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})    
# collect product plots into rows
plots_dim2 = do.call("grid.arrange", c(sample_plots_dim2, ncol=1, top="k=2"))
plots_dim10 = do.call("grid.arrange", c(sample_plots_dim10, ncol=1, top="k=10"))
plots_dim25 = do.call("grid.arrange", c(sample_plots_dim25, ncol=1, top="k=25"))
plots_dim79 = do.call("grid.arrange", c(sample_plots_dim79, ncol=1, top="k=79"))
plots_dim261 = do.call("grid.arrange", c(sample_plots_dim261, ncol=1, top="k=261"))
plots_dim392 = do.call("grid.arrange", c(sample_plots_dim392, ncol=1, top="k=392"))
plots_dim784 = do.call("grid.arrange", c(sample_plots_dim784, ncol=1, top="k=784"))
# combine rows and plot the grid of products with varying k
grid.arrange(plots_dim2, plots_dim10, plots_dim25, plots_dim79, plots_dim261, 
             plots_dim392, plots_dim784, ncol=7) 
```
