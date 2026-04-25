using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Shared.Dtos.PaymentDto
{
    public class PaymentKeyResponse
    {
        public string Token { get; set; }

        [JsonPropertyName("profile_id")]
        public int ProfileId { get; set; }

        public int Expiration { get; set; }

        public bool Success { get; set; }
    }

}
