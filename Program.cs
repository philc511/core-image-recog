using System;

namespace core_image_recog
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] w = {1.0d, 2.0d, 3.0d};
            double[] x = {3.0d, 2.0d, 1.0d};
            //Console.WriteLine(Output(w, x, -1));
            Console.WriteLine(Utils.Activate(-10000.0d));
            Console.WriteLine(Utils.Activate(-0.5d));
            Console.WriteLine(Utils.Activate(0.0d));
            Console.WriteLine(Utils.Activate(0.5d));
            Console.WriteLine(Utils.Activate(10000.0d));
        }




    }
}
