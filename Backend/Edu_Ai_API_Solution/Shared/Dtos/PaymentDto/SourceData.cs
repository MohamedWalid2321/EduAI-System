using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.PaymentDto
{
    public class SourceData
    {
        public string pan { get; set; }
        public string type { get; set; }
        public string sub_type { get; set; }
    }
}
