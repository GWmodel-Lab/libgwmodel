#' Geographically Weighted Principle Components Analysis
#' 
#' Calculate principle components according to GWPCA algorithm.
#' 
#' @param formula Formula used to envaluate principle components with no response (dependent) variables.
#' @param data Spatial*DataFrame as data source.
#' @param bw Spatial bandwidth.
#' @param adaptive True if use adaptive bandwith, or False if use fixed bandwidth.
#' @param kernel Kernel type. Valid values are "gaussian", "bisquare"
#' @param longlat True if geographical coordinates, else False.
#' @param k Number of principle components to be kept.
#' 
#' @return List descripting result of GWPCA algorithm.
#' 
#' @export
gwpca <- function(formula, data, bw, adaptive = T, kernel = "gaussian", longlat = F, k = 2) {
    variables <- attr(terms(formula), which = "variables")
    var.names <- as.character(variables)[-1]
    x <- data@data[var.names]
    points <- data@coords
    .c.gwpca(x, points, var.names, bw, adaptive, kernel, longlat, k)
}