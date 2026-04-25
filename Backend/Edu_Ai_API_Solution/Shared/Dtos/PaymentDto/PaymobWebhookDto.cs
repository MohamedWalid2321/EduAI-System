using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.PaymentDto
{
    public class PaymobWebhookDto
    {
        public WebhookObj obj { get; set; }
        public string hmac { get; set; }

        public bool Status { get; set; }

        public string Id { get; set; }
        public string type { get; set; }
        
    }
}
