using System;
using MathNet.Numerics.LinearAlgebra;

namespace core_image_recog
{
    public class Network
    {
        private Layer[] layers;

        public Network(int[] connections)
        {
            layers = new Layer[connections.Length-1];

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(connections[i+1], connections[i]);
            }

        }

        public double[] Output(double[] x)
        {
            var prevVec = Vector<double>.Build.Dense(x);
            for (int i = 0; i < layers.Length; i++)
            {
                prevVec = layers[i].A(prevVec);
            }

            return prevVec.ToArray();
        }
    }
}