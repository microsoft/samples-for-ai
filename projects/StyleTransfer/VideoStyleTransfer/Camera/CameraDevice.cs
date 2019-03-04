using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DirectShowLib;

namespace VideoStyleTransfer.Camera
{
    public class CameraDevice 
    {
        public int Index { get; set; }
        public string Name { get; set; }
        public Guid Id { get; set; }
        public CameraDevice(int index, string name, Guid id)
        {
            this.Index = index;
            this.Name = name;
            this.Id = id;
        }


        public override String ToString()
        {
            return String.Format("{0}, {1}, {2}", Index, Name, Id.ToString());
        }

        public static CameraDevice[] GetCameraDevice()
        {
            DsDevice[] _devices = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);
            CameraDevice[] cds = new CameraDevice[_devices.Length];
            for (int i = 0; i< _devices.Length; i++)
            {
                cds[i] = new CameraDevice(i, _devices[i].Name, _devices[i].ClassID);
            }
            return cds;
        }
        
    }
    
}
