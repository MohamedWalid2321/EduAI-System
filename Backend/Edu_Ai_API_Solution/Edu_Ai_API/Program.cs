using DomainLayer.Contracts;
using DomainLayer.Models;
using Edu_Ai_API.CustomMiddleWares;
using Edu_Ai_API.Factories;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using persistenceLayer;
using persistenceLayer.Data;
using persistenceLayer.Repository;
using ServiceAbstractionLayer;
using ServiceLayer;
using ServiceLayer.Services;
using Shared.ErrorModels;
using StackExchange.Redis;

namespace Edu_Ai_API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.

            builder.Services.AddControllers();
            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();
			builder.Services.AddIdentity<ApplicationUser, IdentityRole>()
	        .AddEntityFrameworkStores<LmsDBContext>()
	        .AddDefaultTokenProviders();
			builder.Services.AddInfrastructureServices(builder.Configuration);
            builder.Services.AddApplicationServices();


			builder.Services.Configure<ApiBehaviorOptions>((options) =>
            { 
                options.InvalidModelStateResponseFactory = ApiResponseFactory.GenerateApiValidationResponse;
            });

			
			builder.Services.AddCors(options =>
			{
				options.AddDefaultPolicy(builder =>
				{
					builder.AllowAnyOrigin()
						   .AllowAnyMethod()
						   .AllowAnyHeader();
				});
			});



			var app = builder.Build();


            

            app.MapGet("/set/", async ([FromServices] ICacheService cache) =>
            {
                await cache.SetAsync("mykey", "Helloo", TimeSpan.FromMinutes(2));
                return await cache.GetAsync("mykey");
            });

            app.MapGet("/del/", async ([FromServices] ICacheService cache) =>
            {
                await cache.RemoveAsync("mykey");
                
            });




            app.UseMiddleware<CustomExceptionHandlerMiddleWare>();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();
			app.UseCors();

			app.UseAuthentication();

			app.UseAuthorization();

            app.MapControllers();

            app.Run();
        }
    }
}
