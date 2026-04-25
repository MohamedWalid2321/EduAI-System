using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.PaymentDto
{
    public class WebhookResultDto
    {
        public bool Success { get; set; }
        public int OrderId { get; set; }
        public long TransactionId { get; set; }
        
    }
}
