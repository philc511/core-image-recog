using System;
using System.Collections.Generic;

namespace core_image_recog
{
    public class Layer
    {
        private int size;
        private int numInputs;
        private Neuron[] neurons;

        public Layer(int size, int numInputs)
        {
            this.size = size;
            this.numInputs = numInputs;

            this.neurons = new Neuron[size];
            for (int i = 0; i < size; i++)
            {
                this.neurons[i] = new Neuron(numInputs);
            }
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

    }
}