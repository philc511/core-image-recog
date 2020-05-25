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

        private Vector<double> deltaBias;
        private Matrix<double> deltaWeight;

        public Vector<double> A {get; set;}
        public Vector<double> Z {get; set;}

        public Vector<double> Delta {get; set;}

        public Layer(int size, int numInputs)
        {
            this.size = size;
            this.numInputs = numInputs;

            w = Matrix<double>.Build.Dense(size, numInputs, 1d/numInputs);
            b = Vector<double>.Build.Dense(size, -1d);

            deltaWeight = Matrix<double>.Build.Dense(size, numInputs, 0d);
            deltaBias = Vector<double>.Build.Dense(size, 0d);

        }
        public void FeedForward(Vector<double> x)
        {
            Z = w * x + b;
            A = Z.Map(a => Utils.Sigma(a));
        }

        public void ComputeDelta(Vector<double> v)
        {
            Delta = v.PointwiseMultiply(Z.Map(a => Utils.SigmaDash(a)));
        }

        public Vector<double> GetWTransposeDelta()
        {
            return w.Transpose() * Delta;
        }

        public void AdjustDeltaSums(Vector<double> aPrev)
        {
            deltaBias += Delta;
            deltaWeight += Delta.ToColumnMatrix() * aPrev.ToRowMatrix();
        }

        public void GradDesc(double eta, int m) {
            var factor = eta / (1d * m);
            w = w - deltaWeight.Multiply(factor);
            b = b - deltaBias.Multiply(factor);

            deltaWeight.Clear();
            deltaBias.Clear();
        }
    }
}