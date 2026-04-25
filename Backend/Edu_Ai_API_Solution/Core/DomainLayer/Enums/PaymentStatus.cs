using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Enums
{
    public enum PaymentStatus
    {
        Pending = 0,     
        Paid,        
        Failed,      
        Cancelled,   
        Refunded     
    }
}
