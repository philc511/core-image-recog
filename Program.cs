using System;
using MNIST.IO;

namespace core_image_recog
{
    class Program
    {
        static void Main(string[] args)
        {

            var data = FileReaderMNIST.LoadImagesAndLables(
                    "../../data/train-labels-idx1-ubyte.gz",
                    "../../data/train-images-idx3-ubyte.gz");
            // foreach(var i in data)
            // {
            //     //Console.WriteLine(i.AsDouble());
            //     //Console.WriteLine(i.Label.ToString());
            // }
            var network = new Network(new int[]{3,5,1});

            double[] x = {3.0d, 2.0d, 1.0d};
            double[] result = network.Output(x);
            foreach (var d in result) 
            {
                Console.WriteLine(d);
            }
            Console.WriteLine(Utils.Activate(5*Utils.Activate(6d)));
        }




    }
}
