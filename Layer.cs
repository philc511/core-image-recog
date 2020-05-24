using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace core_image_recog
{
    public class Layer
    {
        private int size;
        private int numInputs;

        private Matrix<double> w;

        private Vector<double> b;

        public Layer(int size, int numInputs)
        {
            this.size = size;
            this.numInputs = numInputs;

            w = Matrix<double>.Build.Dense(size, numInputs, 1d/numInputs);
            b = Vector<double>.Build.Dense(size, -1d);
        }
        public Vector<double> A(Vector<double> x)
        {
            var r = w * x + b;
            return (r).Map(a => Utils.Activate(a));
        }
    }
}