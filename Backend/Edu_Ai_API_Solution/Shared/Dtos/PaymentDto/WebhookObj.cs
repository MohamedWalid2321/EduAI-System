using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.PaymentDto
{
    public class WebhookObj
    {
        public long id { get; set; }
        public bool pending { get; set; }
        public int amount_cents { get; set; }
        public bool success { get; set; }
        public bool is_auth { get; set; }
        public bool is_capture { get; set; }
        public bool is_standalone_payment { get; set; }
        public bool is_voided { get; set; }
        public bool is_refunded { get; set; }
        public bool is_3d_secure { get; set; }

        public int integration_id { get; set; }
        public int profile_id { get; set; }

        public bool has_parent_transaction { get; set; }

        public OrderObj order { get; set; }

        public string created_at { get; set; }
        public string currency { get; set; }

        public int owner { get; set; }

        public bool error_occured { get; set; }

        public SourceData source_data { get; set; }
    }
}
