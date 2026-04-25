using DomainLayer.Contracts;
using DomainLayer.Models;
using Edu_Ai_API.CustomMiddleWares;
using Edu_Ai_API.Factories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.OpenApi.Models;
using persistenceLayer;
using persistenceLayer.Data;
using persistenceLayer.Repository;
using PresentationLayer.Authorization;
using Serilog;
using ServiceAbstractionLayer;
using ServiceLayer;
using ServiceLayer.Services;
using Shared.ErrorModels;
using StackExchange.Redis;
using System.Diagnostics;

namespace Edu_Ai_API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Configure Serilog
            builder.Host.UseSerilog((context, configuration) =>
                configuration.ReadFrom.Configuration(context.Configuration));

            // Add services to the container.
            builder.Services.AddControllers();
            
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen(options =>
            {
                options.SwaggerDoc("v1", new OpenApiInfo
                {
                    Title = "Lumino API",
                    Version = "v1",
                    Description = "Learning Management System API",
                    Contact = new OpenApiContact
                    {
                        Name = "Lumina Team",
                        Email = "support@lumino.com"
                    }
                });

                // JWT Authentication
                options.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
                {
                    Name = "Authorization",
                    Type = SecuritySchemeType.Http,
                    Scheme = "Bearer",
                    BearerFormat = "JWT",
                    In = ParameterLocation.Header,
                    Description = "Enter your JWT token"
                });

                options.AddSecurityRequirement(new OpenApiSecurityRequirement
                {
                    {
                        new OpenApiSecurityScheme
                        {
                            Reference = new OpenApiReference
                            {
                                Type = ReferenceType.SecurityScheme,
                                Id = "Bearer"
                            }
                        },
                        Array.Empty<string>()
                    }
                });
            });

            builder.Services.AddIdentity<ApplicationUser, ApplicationRole>()
                .AddEntityFrameworkStores<LmsDBContext>()
                .AddDefaultTokenProviders();

            builder.Services.AddInfrastructureServices(builder.Configuration);
            builder.Services.AddApplicationServices(builder.Environment);

            // Authorization Services
            builder.Services.AddSingleton<IAuthorizationPolicyProvider, PermissionAuthorizationPolicyProvider>();
            builder.Services.AddScoped<IAuthorizationHandler, PermissionAuthorizationHandler>();

            builder.Services.Configure<ApiBehaviorOptions>((options) =>
            {
                options.InvalidModelStateResponseFactory = ApiResponseFactory.GenerateApiValidationResponse;
            });

			builder.Services.AddCors(options =>
			{
				options.AddPolicy("AllowAngular", corsBuilder =>
				{
					corsBuilder.WithOrigins(
                     "http://localhost:4200", 
                     "https://localhost:4200",  
                     "http://localhost:4201",
                     "https://localhost:4201"
        )
        .AllowAnyMethod()
        .AllowAnyHeader()
        .AllowCredentials(); 
				});
			});
			// Configure form options
			builder.Services.Configure<FormOptions>(options =>
			{
				options.MultipartBodyLengthLimit = 524288000; // 500 MB
				options.ValueLengthLimit = int.MaxValue;
				options.MultipartHeadersLengthLimit = int.MaxValue;
			});

			var app = builder.Build();

			// CORS must be FIRST to handle preflight requests
			app.UseCors("AllowAngular");

			// Add Serilog request logging
			app.UseSerilogRequestLogging();

			// Then exception handler
			app.UseMiddleware<CustomExceptionHandlerMiddleWare>();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI(c =>
                {
                    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Lumino API V1");
                    
                    // c.RoutePrefix = string.Empty;
                });
            }

            app.UseHttpsRedirection();
            app.UseAuthentication();
            app.UseAuthorization();
            app.MapControllers();
            
            app.Run();
            
        }
    }
}
