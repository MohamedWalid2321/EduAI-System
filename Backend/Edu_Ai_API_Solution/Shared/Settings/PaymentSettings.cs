using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Settings
{
    
    public class PaymentSettings
    {
        public string ApiKey { get; set; }

        public string IntegrationId { get; set; }

        public string IframeId { get; set; }

        public string HmacSecret { get; set; }

        public string BaseUrl { get; set; }
    }
    
}
