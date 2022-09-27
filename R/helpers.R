curve2 <- function(func, from, to, col = "black", lty = 1, lwd = 1, add = FALSE, xlab = NULL, ylab = NULL, xlim = NULL, ylim = NULL, main = NULL, ...) {
	f <- function(x) func(x)
	curve(expr = f, from = from, to = to, col = col, lty = lty, lwd = lwd, add = add, xlab = xlab, ylab = ylab, xlim = xlim, ylim = ylim, main = main, ... = ...)
}
