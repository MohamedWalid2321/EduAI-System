using DomainLayer.Contracts;
using DomainLayer.Enums;
using DomainLayer.Models;
using Microsoft.Extensions.Options;
using Newtonsoft.Json;
using ServiceAbstractionLayer;
using Shared.Dtos.PaymentDto;
using Shared.Settings;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
    public class PaymobService : IPaymobService
    {

        private readonly HttpClient _httpClient;
        private readonly PaymentSettings _settings;


        public PaymobService(HttpClient httpClient, IOptions<PaymentSettings> options)
        {
            _httpClient = httpClient;
            _settings = options.Value;

        }

        public async Task<string> CreatePaymentUrlAsync(CreatePaymentRequestDto request, CancellationToken cancellationToken = default)
        {
            var token = await GetAuthToken(cancellationToken);

            var orderId = await CreateOrder(token, request, cancellationToken);

            var paymentKey = await GetPaymentKey(token, orderId, request, cancellationToken);

            var url = $"{_settings.BaseUrl}/acceptance/iframes/{_settings.IframeId}?payment_token={paymentKey}";

            return url;
        }

        private async Task<string> GetAuthToken(CancellationToken cancellationToken = default)
        {
            var response = await _httpClient.PostAsJsonAsync(
                $"{_settings.BaseUrl}/auth/tokens",
                new { api_key = _settings.ApiKey }, cancellationToken);

            if (!response.IsSuccessStatusCode)
                throw new Exception("Auth failed");

            var data = await response.Content.ReadFromJsonAsync<AuthResponse>(cancellationToken: cancellationToken);

            return data.token;
        }

        private async Task<int> CreateOrder(string token, CreatePaymentRequestDto request, CancellationToken cancellationToken = default)
        {
            var response = await _httpClient.PostAsJsonAsync(
                $"{_settings.BaseUrl}/ecommerce/orders",
                new
                {
                    auth_token = token,
                    delivery_needed = false,
                    amount_cents = (int)(request.Amount * 100),
                    currency = request.Currency,
                    merchant_order_id = request.OrderId.ToString(),

                }, cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new Exception($"Payment key failed: {error}");
            }



            var data = await response.Content.ReadFromJsonAsync<OrderResponse>(cancellationToken: cancellationToken);

            return data.Id;
        }


        private BillingData BuildBillingData(StudentBillingDto student)
        {
            return new BillingData
            {
                FirstName = student.FirstName,
                LastName = student.LastName,
                Email = student.Email,
                PhoneNumber = student.PhoneNumber,
                //City = student.City ?? "Cairo",
                Country = "EG"
            };
        }

        private async Task<string> GetPaymentKey(string token, int orderId, CreatePaymentRequestDto request, CancellationToken cancellationToken = default)
        {
            if (request.Student == null)
                throw new Exception("Student data is required");

            var billing = BuildBillingData(request.Student);



            var response = await _httpClient.PostAsJsonAsync(
                $"{_settings.BaseUrl}/acceptance/payment_keys",
                new
                {
                    auth_token = token,
                    amount_cents = (int)(request.Amount * 100),
                    expiration = 3600,
                    order_id = orderId,


                    billing_data = new
                    {
                        first_name = billing.FirstName,
                        last_name = billing.LastName,
                        email = billing.Email,
                        phone_number = billing.PhoneNumber,

                        apartment = "NA",
                        floor = "NA",
                        street = "NA",
                        building = "NA",

                        city = billing.City,
                        country = billing.Country,
                        postal_code = "NA",
                        state = "NA"
                    },
                    merchant_order_id = request.OrderId.ToString(),
                    currency = request.Currency,
                    integration_id = int.Parse(_settings.IntegrationId)
                }, cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new Exception($"Payment key failed: {error}");
            }

            var data = await response.Content.ReadFromJsonAsync<PaymentKeyResponse>(cancellationToken: cancellationToken);

            return data.Token;
        }



        public string CalculateHmac(WebhookObj obj)
        {
            
            var values = new List<string>
            {
                obj.amount_cents.ToString(),
                obj.created_at,
                obj.currency,
                obj.error_occured.ToString().ToLower(),
                obj.has_parent_transaction.ToString().ToLower(),
                obj.id.ToString(),
                obj.integration_id.ToString(),
                obj.is_3d_secure.ToString().ToLower(),
                obj.is_auth.ToString().ToLower(),
                obj.is_capture.ToString().ToLower(),
                obj.is_refunded.ToString().ToLower(),
                obj.is_standalone_payment.ToString().ToLower(),
                obj.is_voided.ToString().ToLower(),
                obj.order?.id.ToString(),
                obj.owner.ToString(),
                obj.pending.ToString().ToLower(),
                obj.source_data?.pan,
                obj.source_data?.sub_type,
                obj.source_data?.type,
                obj.success.ToString().ToLower()
            };

            var concatenatedString = string.Concat(values);

            using (var hmac = new HMACSHA512(Encoding.UTF8.GetBytes(_settings.HmacSecret)))
            {
                var hashBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes(concatenatedString));

                StringBuilder sb = new StringBuilder();
                foreach (var b in hashBytes)
                {
                    sb.Append(b.ToString("x2"));
                }
                return sb.ToString();
            }
        }
    }

}
