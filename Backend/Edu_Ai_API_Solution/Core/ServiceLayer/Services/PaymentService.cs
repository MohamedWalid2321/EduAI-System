using DomainLayer.Enums;
using Newtonsoft.Json;
using ServiceLayer.Specifications.FeeSpecifications;
using ServiceLayer.Specifications.PaymentSpecifications;
using Shared.Dtos.PaymentDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
    public class PaymentService(IUnitOfWork unitOfWork , UserManager<ApplicationUser> userManager,IPaymobService paymobService) : IPaymentService
    {
        private readonly IUnitOfWork _unitOfWork = unitOfWork;
        private readonly UserManager<ApplicationUser> _userManager = userManager;
        private readonly IPaymobService _paymobService = paymobService;

        public async Task<string> CreatePaymentAsync(string studentId, CancellationToken cancellationToken = default)
        {
            
            var feeRepo = _unitOfWork.GetRepository<Fee, int>();
            var paymentRepo = _unitOfWork.GetRepository<Payment, int>();

            var student = await _userManager.FindByIdAsync(studentId);

            if (student == null)
                throw new UserNotFound(studentId);
            

            var spec = new PaidFeeSpecification(studentId, student.AcademicYearId);


            var exists = await paymentRepo.GetAllAsync(spec, cancellationToken);

            if (exists.Any())
                throw new Exception("Already paid");

            var feeSpecification = new FeeSpecifications(student.AcademicYearId, student.DepartmentId);

            var fees = await feeRepo.GetAllAsync(feeSpecification, cancellationToken);

            var totalAmount = fees.Sum(f => f.Amount);

            var payment = new Payment
            {
                StudentId = studentId,
                AcademicYearId = student.AcademicYearId,
                Amount = totalAmount,
                Status = PaymentStatus.Pending,
                CreatedAt = DateTime.UtcNow,
                Currency = "EGP",
                TransactionId = "NA"
            };

            await paymentRepo.AddAsync(payment, cancellationToken);
            try
            {
                await _unitOfWork.SaveChangesAsync(cancellationToken);
            }
            catch (Exception ex)
            {
                var inner = ex.InnerException?.Message;
                throw new Exception(inner ?? ex.Message);
            }

            var dto = new CreatePaymentRequestDto();

            // 3. call Paymob
            dto.Amount = totalAmount;
            dto.OrderId = payment.Id.ToString();
            dto.Currency = payment.Currency;
            dto.Student = new()
            {
                FirstName = string.IsNullOrEmpty(student.FirstName) ? "NA" : student.FirstName,
                LastName = string.IsNullOrEmpty(student.LastName) ? "NA" : student.LastName,
                Id = student.Id,
                Email = string.IsNullOrEmpty(student.Email) ? "NA" : student.Email,
                PhoneNumber = string.IsNullOrEmpty(student.PhoneNumber) ? "NA" : student.PhoneNumber,
            };

            var url = await _paymobService.CreatePaymentUrlAsync(dto, cancellationToken);

            return url;
        }

        

        public async Task<WebhookResultDto> HandleWebhookAsync(string body, string sentHmac, CancellationToken cancellationToken = default)
        {
            var data = JsonConvert.DeserializeObject<PaymobWebhookDto>(body);

            if (data?.obj == null)
                throw new Exception("Invalid webhook payload");

            var calculatedHmac = _paymobService.CalculateHmac(data.obj);    

            if (!string.Equals(calculatedHmac, sentHmac, StringComparison.OrdinalIgnoreCase))
                throw new Exception("Invalid HMAC");
           
            var paymentRepo = _unitOfWork.GetRepository<Payment, int>();

            var orderId = data.obj.order.merchant_order_id;

            var spec = new PaymentByOrderIdSpec(orderId);

            var payment = await paymentRepo.GetFirstOrDefaultAsync(spec, cancellationToken);

            if (payment == null)
                throw new Exception("Payment not found");

            if (payment.Status == PaymentStatus.Paid)
                return new WebhookResultDto
                {
                    OrderId = data.obj.order.id,
                    TransactionId = data.obj.id,
                    Success = true
                };

            bool isSuccess = data.obj.success;
            //bool isPaid = data.obj.is_paid;

            payment.TransactionId = data.obj.id.ToString();
            payment.Status = (isSuccess)
                ? PaymentStatus.Paid
                : PaymentStatus.Failed;

            _unitOfWork.GetRepository<Payment, int>().Update(payment);
            await _unitOfWork.SaveChangesAsync(cancellationToken);

            return new WebhookResultDto
            {
                OrderId = data.obj.order.id,
                TransactionId = data.obj.id,
                Success = payment.Status == PaymentStatus.Paid
            };
        }
    }
}
