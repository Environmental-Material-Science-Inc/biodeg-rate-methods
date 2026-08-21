# Oracle fit for the Module 3 validation: mgcv tensor-product P-spline by REML.
# Args: data.csv query.csv out.csv k1 k2 k3
# data.csv cols: e,n,t,y   query.csv cols: e,n,t   -> out.csv adds column pred.
# This is the reference implementation named in degradation_methods_mathematical_reference.md
# Appendix B.8: gam(y ~ te(s1,s2,t, bs="ps"), method="REML").
args <- commandArgs(trailingOnly = TRUE)
datacsv <- args[1]; querycsv <- args[2]; outcsv <- args[3]
k1 <- as.integer(args[4]); k2 <- as.integer(args[5]); k3 <- as.integer(args[6])
suppressMessages(library(mgcv))

d <- read.csv(datacsv)
q <- read.csv(querycsv)
fit <- gam(y ~ te(e, n, t, bs = "ps", k = c(k1, k2, k3)), data = d, method = "REML")
q$pred <- as.numeric(predict(fit, newdata = q))
write.csv(q, outcsv, row.names = FALSE)

edf <- sum(fit$edf)
cat(sprintf("EDF=%.4f REML=%.4f n=%d\n", edf, fit$gcv.ubre, nrow(d)), file = stderr())
