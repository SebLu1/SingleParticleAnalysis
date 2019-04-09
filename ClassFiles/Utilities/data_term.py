# TODO: The adjoint of the Fourier transform is taken to be the inverse, which is not exactly correct when using rfft.
class DataTerm(odl.solvers.Functional):
    def __init__(self, domain):
        super(DataTerm, self).__init__(domain)  # , range=odl.RealNumbers())

    def data_gradient(self, prod_elem):

        image = prod_elem[0]
        kernel = prod_elem[1]
        data = prod_elem[2]

        fourier_data = np.fft.fftshift(np.fft.rfftn(image))

        grad = np.multiply(kernel, fourier_data) - data

        return np.fft.ifftshift(np.fft.irfftn(grad))

    def _call(self, prod_elem):
        return l2(self.data_gradient(prod_elem))  # optimal funtional value

    # For performance OrbitLossGradientOperator should maybe defined outside of orbitLoss?
    @property
    def gradient(self):
        class DataTermGrad(odl.Operator):
            def __init__(self, domain, outer_instance):
                super(OrbitLossGradientOperator, self).__init__(domain=domain,
                                                                range=domain)  # , range=odl.RealNumbers())
                self.outer_instance = outer_instance

            def _call(self, prod_elem):
                x = prod_elem[0]  # Ground truth
                y = prod_elem[1]  # Data
                vec_x = np.zeros_like(
                    x)  # Lets set the dervative wrt ground truth-part to zero. It will not propagate back to network params.
                theta = self.outer_instance.localReg(prod_elem)[0]  # optimal angle
                vec_y = y - skimage_rot(x, theta, order=3)
                vec = self.domain.element([vec_x, vec_y])
                return vec

        return OrbitLossGradientOperator(self.domain, self)