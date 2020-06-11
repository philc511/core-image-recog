using System;
using MathNet.Numerics.LinearAlgebra;

namespace NnetLib
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

        public Vector<double> FeedForward(Vector<double> x)
        {
            var prevVec = x;
            foreach (var layer in layers)
            {
                layer.FeedForward(prevVec);
                prevVec = layer.A;
            }
            return layers[layers.Length-1].A;
        }

        public void BackProp(Vector<double> y)
        {
            layers[layers.Length-1].ComputeDelta(y);
            for (int i = layers.Length - 2 ; i>=0 ; i--)
            {
                layers[i].ComputeDelta(layers[i+1].GetWTransposeDelta());
            }
        }

        public void AdjustDeltaSums(Vector<double> x)
        {
            var prevVec = x;
            foreach (var layer in layers)
            {
                layer.AdjustDeltaSums(prevVec);
                prevVec = layer.A;
            }
        }
        public void GradDesc(double eta, int m)
        {
            foreach (var layer in layers)
            {
                layer.GradDesc(eta, m);
            }
        }
    }
}