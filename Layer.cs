using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace core_image_recog
{
    public class Layer
    {
        private int size;
        private int numInputs;
        private Neuron[] neurons;

        private Matrix<double> w;


        public Layer(int size, int numInputs)
        {
            this.size = size;
            this.numInputs = numInputs;

            this.neurons = new Neuron[size];
            for (int i = 0; i < size; i++)
            {
                this.neurons[i] = new Neuron(numInputs);
            }

            w = Matrix<double>.Build.Random(size, numInputs);
        }
        public double[] Output(double[] x)
        {
            var output = new double[size];
            for (int i = 0; i < size; i++)
            {
                output[i] = this.neurons[i].Output(x);
            }
            return output;
        }

        public Vector<double> A(Vector<double> x)
        {
            return (w * x).Map(a => Utils.Activate(a));
        }
    }
}